// Identical finite workload for BaM and CoPilotIO; one NVMe command per operation.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <memory>
#include <sched.h>
#include <ctrl.h>
#include <page_cache.h>
#ifdef SWARMIO_COPILOT
#include <copilot.h>
#endif

struct Config {
    uint64_t threads=64, blk_size=64, reqs=1, pages=4096, page_size=512;
    uint64_t num_blks=2097152, queue_depth=512, num_queues=1;
    int gpu=0, base_core=0, cores=4;
    bool random=false;
    bool host_notify=false;
    bool relay_doorbells=false;
    bool probe_only=false;
};
static uint64_t number(const std::string& s) {
    size_t n=0;
    if (s.empty() || s[0]=='-') throw std::runtime_error("invalid unsigned number");
    auto v=std::stoull(s,&n);
    if(n!=s.size()) throw std::runtime_error("invalid number suffix");
    return v;
}
static Config parse(int argc,char** argv) {
    Config c;
    for(int i=1;i<argc;i++) {
        std::string a=argv[i]; auto eq=a.find('=');
        if(eq==std::string::npos) throw std::runtime_error("expected --name=value");
        auto k=a.substr(0,eq), s=a.substr(eq+1);
        if(k=="--probe_only") {
            if(s!="true" && s!="false") throw std::runtime_error("invalid probe boolean");
            c.probe_only=s=="true"; continue;
        }
        if(k=="--random") { if(s!="true" && s!="false") throw std::runtime_error("invalid boolean"); c.random=s=="true"; continue; }
        if(k=="--notify_memory") {
            if(s!="host" && s!="gdr") throw std::runtime_error("invalid notification memory");
            c.host_notify=s=="host"; continue;
        }
        if(k=="--doorbell_mode") {
            if(s!="mmio" && s!="relay") throw std::runtime_error("invalid doorbell mode");
            c.relay_doorbells=s=="relay"; continue;
        }
        auto v=number(s);
        if(k=="--threads") c.threads=v;
        else if(k=="--blk_size") c.blk_size=v;
        else if(k=="--reqs") c.reqs=v;
        else if(k=="--pages") c.pages=v;
        else if(k=="--page_size") c.page_size=v;
        else if(k=="--num_blks") c.num_blks=v;
        else if(k=="--queue_depth") c.queue_depth=v;
        else if(k=="--num_queues") c.num_queues=v;
        else if(k=="--gpu" && v<=1024) c.gpu=v;
        else if(k=="--copilot_base_core" && v<CPU_SETSIZE) c.base_core=v;
        else if(k=="--copilot_cores" && v>0 && v<=64) c.cores=v;
        else if(k=="--access_type" && v==0) {}
        else if(k=="--n_ctrls" && v==1) {}
        else throw std::runtime_error("unsupported option: "+a);
    }
    if(!c.threads || c.threads%64 || c.threads>0x7fffffff || c.blk_size!=64 || !c.reqs ||
       c.reqs>UINT64_MAX/c.threads || c.pages<c.threads ||
       (c.page_size!=512 && c.page_size!=4096) || c.num_blks<c.threads ||
       c.num_blks>UINT64_MAX/c.page_size || !c.num_queues || c.num_queues>64 ||
       c.queue_depth<2 || c.queue_depth>1024 || (c.queue_depth&(c.queue_depth-1)))
        throw std::runtime_error("invalid workload geometry (pages must cover threads)");
    return c;
}

// No admin reset, queue creation, or NVMe read/write commands in this check.
static int probe_mapping(const Config& c) {
    int fd=open("/dev/libnvm0",O_RDWR);
    if(fd<0) throw error("probe: failed to open /dev/libnvm0");
    nvm_ctrl_t* raw=nullptr;
    int status=nvm_ctrl_init(&raw,fd);
    close(fd);
    if(!nvm_ok(status)) throw error(std::string("probe: controller mmap: ")+nvm_strerror(status));
    std::unique_ptr<nvm_ctrl_t,decltype(&nvm_ctrl_free)> ctrl(raw,&nvm_ctrl_free);
    printf("Probe stage: doorbell %s\n", c.relay_doorbells ? "relay" : "mmio"); fflush(stdout);
    DoorbellMapping doorbells;
    doorbells.init(ctrl.get(),c.num_queues,c.relay_doorbells);
    if(!c.relay_doorbells) {
        void* alias=nullptr;
        auto err=cudaHostGetDevicePointer(&alias,(void*)ctrl->mm_ptr,0);
        if(err!=cudaSuccess) throw error(std::string("probe: BAR CUDA alias: ")+cudaGetErrorString(err));
    }
    printf("Probe stage: host DMA mapping\n"); fflush(stdout);
    auto host=createDma(ctrl.get(),65536);
    printf("Probe stage: GPU P2P DMA mapping\n"); fflush(stdout);
    auto gpu=createDma(ctrl.get(),65536,c.gpu);
    if(!host->n_ioaddrs || !gpu->n_ioaddrs) throw error("probe: empty DMA mapping");
#ifdef SWARMIO_COPILOT
    printf("Probe stage: notification %s\n",c.host_notify ? "host" : "gdr"); fflush(stdout);
    CoPilotNotify notify;
    notify.init(c.host_notify);
#endif
    cuda_err_chk(cudaDeviceSynchronize());
    printf("Probe OK: registration and DMA mappings only; no NVMe I/O issued\n");
    return 0;
}

__global__ void read_kernel(Controller** ctrls,page_cache_d_t* pc,
    uint64_t threads,uint64_t reqs,uint64_t range,uint64_t blocks,bool random
#ifdef SWARMIO_COPILOT
    ,copilot_notify_entry** notifies
#endif
) {
    uint64_t tid=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid>=threads) return;
    auto qp=ctrls[0]->d_qps+(tid%ctrls[0]->n_qps);
    uint64_t seed=tid*6364136223846793005ULL+1442695040888963407ULL;
    for(uint64_t i=0;i<reqs;i++) {
        seed^=seed<<13; seed^=seed>>7; seed^=seed<<17;
        uint64_t lba=(random ? seed%range : (tid+i)%range)*blocks;
        nvm_cmd_t cmd;
        uint16_t cid=get_cid(&qp->sq);
        nvm_cmd_header(&cmd,cid,NVM_IO_READ,qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd,pc->prp1[tid],pc->prps ? pc->prp2[tid] : 0);
        nvm_cmd_rw_blks(&cmd,lba,blocks);
        sq_enqueue(&qp->sq,&cmd);
        uint32_t head,head_,pos;
#ifdef SWARMIO_COPILOT
        volatile copilot_notify_entry* notify=notifies[tid%ctrls[0]->n_qps];
        while(!notify[cid].ready) __nanosleep(64);
        // CPU publishes metadata before ready; volatile reads bypass GPU L1.
        __threadfence_system();
        pos=notify[cid].pos; head=notify[cid].loc; head_=notify[cid].cq_head;
        __threadfence_system();
        notify[cid].ready=0;
        __threadfence_system();
#else
        pos=cq_poll(&qp->cq,cid,&head,&head_);
#endif
        if ((static_cast<volatile nvm_cpl_t*>(qp->cq.vaddr)[pos].dword[3] >> 17) != 0) {
            printf("NVMe read error cid=%u\n",unsigned(cid)); asm("trap;");
        }
        cq_dequeue(&qp->cq,pos,&qp->sq,head,head_);
        put_cid(&qp->sq,cid);
    }
}
int main(int argc,char** argv) try {
    auto c=parse(argc,argv);
    cuda_err_chk(cudaSetDevice(c.gpu));
    printf("Doorbell mode: %s\n", c.relay_doorbells ? "relay" : "mmio");
    if(c.probe_only) return probe_mapping(c);
#ifdef SWARMIO_COPILOT
    const char* engine="copilotio";
    Controller ctrl("/dev/libnvm0",1,c.gpu,c.queue_depth,c.num_queues,QUEUE_CPU,QUEUE_CPU,c.relay_doorbells);
#else
    const char* engine="bam";
    Controller ctrl("/dev/libnvm0",1,c.gpu,c.queue_depth,c.num_queues,QUEUE_GPU,QUEUE_GPU,c.relay_doorbells);
#endif
    if(ctrl.n_qps!=c.num_queues || c.page_size%ctrl.blk_size ||
       c.num_blks>ctrl.ns.size/(c.page_size/ctrl.blk_size)) throw std::runtime_error("namespace or queue limit mismatch");
    for(unsigned q=0;q<ctrl.n_qps;q++)
        if(ctrl.h_qps[q]->sq.qs!=c.queue_depth || ctrl.h_qps[q]->cq.qs!=c.queue_depth)
            throw std::runtime_error("controller clamped queue depth");
    std::vector<Controller*> ctrls{&ctrl};
    page_cache_t pc(c.page_size,c.pages,c.gpu,ctrl,64,ctrls);
#ifdef SWARMIO_COPILOT
    std::vector<CoPilotNotify> notifies(c.num_queues);
    std::vector<copilot_notify_entry*> ptrs(c.num_queues);
    for(unsigned q=0;q<c.num_queues;q++) { notifies[q].init(c.host_notify); ptrs[q]=notifies[q].gpu_ptr; }
    copilot_notify_entry** d_ptrs=nullptr;
    cuda_err_chk(cudaMalloc(&d_ptrs,ptrs.size()*sizeof(ptrs[0])));
    cuda_err_chk(cudaMemcpy(d_ptrs,ptrs.data(),ptrs.size()*sizeof(ptrs[0]),cudaMemcpyHostToDevice));
    CoPilotAgent agent;
    printf("Notification memory: %s\n", c.host_notify ? "host" : "gdr");
    agent.start(c.num_queues,ctrl.h_qps,notifies.data(),c.base_core,c.cores);
#endif
    cudaEvent_t begin,end;
    cuda_err_chk(cudaEventCreate(&begin)); cuda_err_chk(cudaEventCreate(&end));
    cuda_err_chk(cudaEventRecord(begin));
    read_kernel<<<c.threads/64,64>>>(pc.pdt.d_ctrls,pc.d_pc_ptr,c.threads,c.reqs,c.num_blks,c.page_size/ctrl.blk_size,c.random
#ifdef SWARMIO_COPILOT
        ,d_ptrs
#endif
    );
    cuda_err_chk(cudaGetLastError());
    cuda_err_chk(cudaEventRecord(end)); cuda_err_chk(cudaEventSynchronize(end));
    float ms=0; cuda_err_chk(cudaEventElapsedTime(&ms,begin,end));
#ifdef SWARMIO_COPILOT
    agent.stop(); cuda_err_chk(cudaFree(d_ptrs));
#endif
    double ops=double(c.threads)*c.reqs;
    printf("Elapsed Time: %.6f Number of Ops: %llu Data Size (bytes): %llu\n",
           double(ms)*1000.0, (unsigned long long)(c.threads*c.reqs),
           (unsigned long long)(c.threads*c.reqs*c.page_size));
    printf("Engine: %s\nOperations: %llu\nElapsed ms: %.6f\nOps/sec: %.6f\nBandwidth bytes/sec: %.6f\n",engine,
           (unsigned long long)(c.threads*c.reqs),ms,ops/(ms/1000.0),ops*c.page_size/(ms/1000.0));
    cuda_err_chk(cudaEventDestroy(begin)); cuda_err_chk(cudaEventDestroy(end));
    return 0;
} catch(const std::exception& e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
