/*
This is the exact kernel code as it appears in the main parser (SerenaParser.h). It was used to generate all training logs. The code is messy and depends on the parser context
It keeps the original shape of the architecture, but the emitter layer was removed.
This file is not meant to be standalone or runnable.
If you want a clean standalone version, see qitnn/cuda-kernels/qitnnRefactored.cu

External symbols from the language/runtime that are referenced here but not included:
- _sr_gpu_init()        : GPU runtime init.
- _sr_gpu_ok            : CUDA availability flag.
- _sr_wmma_ok           : WMMA availability flag.
- SR_CU(expr)           : CUDA call wrapper.
- SR_BLK_M/N/K          : WMMA tile sizes.
- SR_BLOCK_THREADS      : WMMA kernel thread count.
- _Sr_GpuPackA_F16K     : pack A into padded FP16 layout.
- _Sr_GpuPackB_F16K     : pack B into padded FP16 layout.
- _Sr_GpuGemmWmmaK      : WMMA GEMM kernel.
- Sr_GpuGemm(...)       : existing host GEMM entry used in verify.

The goal here is to show the kernel architecture itself with minimal surrounding language code.
*/



/*
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
QITNN core
Qutrit state: psi = a_neg|-1> + a_zero|0> + a_pos|+1>
Forward path: 3x GEMM -> Born normalize -> expected value
Born rule: P(k) = C_k^2 / (C_neg^2 + C_zero^2 + C_pos^2)
Expected value: E = P(+1) - P(-1)
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
*/
__global__ void _Sr_GpuQtsNormalizeK(
    const float* __restrict__ cn, const float* __restrict__ cz,
    const float* __restrict__ cp,
    float* __restrict__ pn, float* __restrict__ pz,
    float* __restrict__ pp, float* __restrict__ ev, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float an = cn[i], az = cz[i], ap = cp[i];
  float qn = an*an, qz = az*az, qp = ap*ap;
  float z = qn + qz + qp;
  float iz = (z > 1e-12f) ? (1.0f / z) : 0.0f;
  qn *= iz; qz *= iz; qp *= iz;
  pn[i] = qn; pz[i] = qz; pp[i] = qp;
  ev[i] = qp - qn;
}

__global__ void _Sr_GpuQtsRandFillK(float* __restrict__ out, int n, unsigned int seed) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  unsigned int s = seed ^ (unsigned int)(i * 2654435761u);
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  out[i] = (float)(s & 0xFFFFu) * (1.0f/32768.0f) - 1.0f;
}

__global__ void _Sr_GpuQtsNormAmplK(
    float* __restrict__ an, float* __restrict__ az,
    float* __restrict__ ap, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float a = an[i], b = az[i], c = ap[i];
  float norm = sqrtf(a*a + b*b + c*c);
  float inv = (norm > 1e-6f) ? (1.0f / norm) : 0.57735f;
  an[i] = a * inv; az[i] = b * inv; ap[i] = c * inv;
}

extern "C" double Sr_GpuQtsBench(int M, int N, int K) {
  _sr_gpu_init();
  if (!_sr_gpu_ok || !_sr_wmma_ok) { std::fprintf(stderr,"[QTS] WMMA not available\n"); return 0.0; }
  const int Mp = (M + SR_BLK_M - 1) & ~(SR_BLK_M - 1);
  const int Np = (N + SR_BLK_N - 1) & ~(SR_BLK_N - 1);
  const int Kp = (K + SR_BLK_K - 1) & ~(SR_BLK_K - 1);
  float *dX=nullptr, *dAn=nullptr, *dAz=nullptr, *dAp=nullptr;
  size_t szX=(size_t)M*K*sizeof(float), szA=(size_t)K*N*sizeof(float);
  SR_CU(cudaMalloc(&dX,szX));
  SR_CU(cudaMalloc(&dAn,szA)); SR_CU(cudaMalloc(&dAz,szA)); SR_CU(cudaMalloc(&dAp,szA));
  _Sr_GpuQtsRandFillK<<<(M*K+255)/256,256>>>(dX, M*K, 0xDEAD0001u);
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAn, K*N, 0xDEAD0002u);
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAz, K*N, 0xDEAD0003u);
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAp, K*N, 0xDEAD0004u);
  _Sr_GpuQtsNormAmplK<<<(K*N+255)/256,256>>>(dAn, dAz, dAp, K*N);
  SR_CU(cudaDeviceSynchronize());
  __half *dXh=nullptr, *dAnh=nullptr, *dAzh=nullptr, *dAph=nullptr;
  size_t szXh=(size_t)Mp*Kp*sizeof(__half), szAh=(size_t)Kp*Np*sizeof(__half);
  SR_CU(cudaMalloc(&dXh,szXh)); SR_CU(cudaMemset(dXh,0,szXh));
  SR_CU(cudaMalloc(&dAnh,szAh)); SR_CU(cudaMemset(dAnh,0,szAh));
  SR_CU(cudaMalloc(&dAzh,szAh)); SR_CU(cudaMemset(dAzh,0,szAh));
  SR_CU(cudaMalloc(&dAph,szAh)); SR_CU(cudaMemset(dAph,0,szAh));
  float *dCn=nullptr, *dCz=nullptr, *dCp=nullptr;
  size_t szC=(size_t)Mp*Np*sizeof(float);
  SR_CU(cudaMalloc(&dCn,szC)); SR_CU(cudaMalloc(&dCz,szC)); SR_CU(cudaMalloc(&dCp,szC));
  float *dPn=nullptr, *dPz=nullptr, *dPp=nullptr, *dEvB=nullptr;
  SR_CU(cudaMalloc(&dPn,szC)); SR_CU(cudaMalloc(&dPz,szC)); SR_CU(cudaMalloc(&dPp,szC)); SR_CU(cudaMalloc(&dEvB,szC));
  dim3 block(SR_BLOCK_THREADS,1,1);
  dim3 grid((Np+SR_BLK_N-1)/SR_BLK_N, (Mp+SR_BLK_M-1)/SR_BLK_M);
  int normN=Mp*Np;
  SR_CU(cudaFuncSetAttribute((const void*)_Sr_GpuGemmWmmaK, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  for(int w=0;w<5;w++){
    _Sr_GpuPackA_F16K<<<(M*K+255)/256,256>>>(dX,dXh,M,K,Kp);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAn,dAnh,K,N,Np);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAz,dAzh,K,N,Np);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAp,dAph,K,N,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAnh,dCn,Mp,Np,Kp,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAzh,dCz,Mp,Np,Kp,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAph,dCp,Mp,Np,Kp,Np);
    _Sr_GpuQtsNormalizeK<<<(normN+255)/256,256>>>(dCn,dCz,dCp,dPn,dPz,dPp,dEvB,normN);
  }
  SR_CU(cudaGetLastError()); SR_CU(cudaDeviceSynchronize());
  cudaEvent_t ev0,ev1;
  SR_CU(cudaEventCreate(&ev0)); SR_CU(cudaEventCreate(&ev1));
  const int runs=20;
  SR_CU(cudaEventRecord(ev0));
  for(int r=0;r<runs;r++){
    _Sr_GpuPackA_F16K<<<(M*K+255)/256,256>>>(dX,dXh,M,K,Kp);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAn,dAnh,K,N,Np);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAz,dAzh,K,N,Np);
    _Sr_GpuPackB_F16K<<<(K*N+255)/256,256>>>(dAp,dAph,K,N,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAnh,dCn,Mp,Np,Kp,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAzh,dCz,Mp,Np,Kp,Np);
    _Sr_GpuGemmWmmaK<<<grid,block>>>(dXh,dAph,dCp,Mp,Np,Kp,Np);
    _Sr_GpuQtsNormalizeK<<<(normN+255)/256,256>>>(dCn,dCz,dCp,dPn,dPz,dPp,dEvB,normN);
  }
  SR_CU(cudaEventRecord(ev1));
  SR_CU(cudaEventSynchronize(ev1));
  float ms=0.f;
  SR_CU(cudaEventElapsedTime(&ms,ev0,ev1));
  double avgMs=(double)ms/runs;
  double qtops=(double)M*(double)N*(double)K/(avgMs*1e9);
  double fp16tflops=qtops*12.0;
  std::fprintf(stderr,"[QTS] %dx%dx%d | %d runs | full pipeline (pack+3xGEMM+normalize)\n",M,N,K,runs);
  std::fprintf(stderr,"[QTS] avg=%.4f ms | %.2f TQ-TOPS (1 QTOP=12 FP16) | %.1f FP16 TFLOPS\n",avgMs,qtops,fp16tflops);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  cudaFree(dX); cudaFree(dAn); cudaFree(dAz); cudaFree(dAp);
  cudaFree(dXh); cudaFree(dAnh); cudaFree(dAzh); cudaFree(dAph);
  cudaFree(dCn); cudaFree(dCz); cudaFree(dCp);
  cudaFree(dPn); cudaFree(dPz); cudaFree(dPp); cudaFree(dEvB);
  return qtops;
}

extern "C" int Sr_GpuQtsVerify(int M, int N, int K) {
  _sr_gpu_init(); if(!_sr_gpu_ok) return -1;
  int totalErrs=0;
  std::fprintf(stderr,"[QTS] === Phase A: normalize micro-tests ===\n");
  {
    const int NT=7;
    float hCn[7]={0.f, 1.f, 0.f, 3.f, 1.f, 5.f, 1.f};
    float hCz[7]={0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 2.f};
    float hCp[7]={1.f, 0.f, 0.f, 4.f, 1.f,-5.f,10.f};
    float eEv[7]={1.f,-1.f,0.f,0.28f,0.f,0.f,99.f/105.f};
    float ePn[7]={0.f,1.f,0.f,9.f/25.f,1.f/3.f,0.5f,1.f/105.f};
    float ePz[7]={0.f,0.f,1.f,0.f,1.f/3.f,0.f,4.f/105.f};
    float ePp[7]={1.f,0.f,0.f,16.f/25.f,1.f/3.f,0.5f,100.f/105.f};
    const char* nm[7]={"pure|+1>","pure|-1>","pure|0>","sup(3,0,4)","equal(1,1,1)","sign(5,0,-5)","bias(1,2,10)"};
    float *dCn,*dCz,*dCp,*dPn,*dPz,*dPp,*dEv;
    SR_CU(cudaMalloc(&dCn,NT*sizeof(float))); SR_CU(cudaMalloc(&dCz,NT*sizeof(float)));
    SR_CU(cudaMalloc(&dCp,NT*sizeof(float))); SR_CU(cudaMalloc(&dPn,NT*sizeof(float)));
    SR_CU(cudaMalloc(&dPz,NT*sizeof(float))); SR_CU(cudaMalloc(&dPp,NT*sizeof(float)));
    SR_CU(cudaMalloc(&dEv,NT*sizeof(float)));
    SR_CU(cudaMemcpy(dCn,hCn,NT*sizeof(float),cudaMemcpyHostToDevice));
    SR_CU(cudaMemcpy(dCz,hCz,NT*sizeof(float),cudaMemcpyHostToDevice));
    SR_CU(cudaMemcpy(dCp,hCp,NT*sizeof(float),cudaMemcpyHostToDevice));
    _Sr_GpuQtsNormalizeK<<<1,256>>>(dCn,dCz,dCp,dPn,dPz,dPp,dEv,NT);
    SR_CU(cudaGetLastError()); SR_CU(cudaDeviceSynchronize());
    float hPn[7],hPz[7],hPp[7],hEv[7];
    SR_CU(cudaMemcpy(hPn,dPn,NT*sizeof(float),cudaMemcpyDeviceToHost));
    SR_CU(cudaMemcpy(hPz,dPz,NT*sizeof(float),cudaMemcpyDeviceToHost));
    SR_CU(cudaMemcpy(hPp,dPp,NT*sizeof(float),cudaMemcpyDeviceToHost));
    SR_CU(cudaMemcpy(hEv,dEv,NT*sizeof(float),cudaMemcpyDeviceToHost));
    for(int t=0;t<NT;t++){
      float dE=hEv[t]-eEv[t]; if(dE<0)dE=-dE;
      float dp=hPn[t]-ePn[t]; if(dp<0)dp=-dp;
      float dz=hPz[t]-ePz[t]; if(dz<0)dz=-dz;
      float dr=hPp[t]-ePp[t]; if(dr<0)dr=-dr;
      float sp=hPn[t]+hPz[t]+hPp[t];
      int ok=(dE<0.002f&&dp<0.002f&&dz<0.002f&&dr<0.002f&&sp>0.999f&&sp<1.001f);
      if(!ok)totalErrs++;
      std::fprintf(stderr,"  Q%d %-16s cn=%.1f cz=%.1f cp=%.1f | E=%.4f(%.4f) P=(%.4f,%.4f,%.4f) sum=%.6f %s\n",
        t+1,nm[t],hCn[t],hCz[t],hCp[t],hEv[t],eEv[t],hPn[t],hPz[t],hPp[t],sp,ok?"PASS":"FAIL");
    }
    cudaFree(dCn);cudaFree(dCz);cudaFree(dCp);
    cudaFree(dPn);cudaFree(dPz);cudaFree(dPp);cudaFree(dEv);
  }
  std::fprintf(stderr,"[QTS] === Phase B: Full pipeline %dx%dx%d ===\n",M,N,K);
  {
    size_t szX=(size_t)M*K*sizeof(float);
    size_t szA=(size_t)K*N*sizeof(float);
    size_t szC=(size_t)M*N*sizeof(float);
    float* hX=(float*)std::malloc(szX);
    float* hAn=(float*)std::malloc(szA);
    float* hAz=(float*)std::malloc(szA);
    float* hAp=(float*)std::malloc(szA);
    float* hCn=(float*)std::malloc(szC);
    float* hCz=(float*)std::malloc(szC);
    float* hCp=(float*)std::malloc(szC);
    float* hEvG=(float*)std::malloc(szC);
    float* hRef=(float*)std::malloc(szC);
    unsigned int _xs=0xDEADBEEFu;
    for(int i=0;i<M*K;i++){
      _xs^=_xs<<13;_xs^=_xs>>17;_xs^=_xs<<5;
      hX[i]=((float)(_xs&0xFFFF)/32768.f)-1.f;
    }
    for(int i=0;i<K*N;i++){
      float rn,rz,rp;
      _xs^=_xs<<13;_xs^=_xs>>17;_xs^=_xs<<5; rn=((float)(_xs&0xFFFF)/32768.f)-1.f;
      _xs^=_xs<<13;_xs^=_xs>>17;_xs^=_xs<<5; rz=((float)(_xs&0xFFFF)/32768.f)-1.f;
      _xs^=_xs<<13;_xs^=_xs>>17;_xs^=_xs<<5; rp=((float)(_xs&0xFFFF)/32768.f)-1.f;
      float s=sqrtf(rn*rn+rz*rz+rp*rp);
      if(s>1e-8f){float is=1.f/s;rn*=is;rz*=is;rp*=is;}
      else{rn=0;rz=0;rp=1;}
      hAn[i]=rn; hAz[i]=rz; hAp[i]=rp;
    }
    Sr_GpuGemm(hX,hAn,hCn,M,N,K);
    Sr_GpuGemm(hX,hAz,hCz,M,N,K);
    Sr_GpuGemm(hX,hAp,hCp,M,N,K);
    float *dCnV,*dCzV,*dCpV,*dPnV,*dPzV,*dPpV,*dEvV;
    int nel=M*N;
    SR_CU(cudaMalloc(&dCnV,szC));SR_CU(cudaMalloc(&dCzV,szC));SR_CU(cudaMalloc(&dCpV,szC));
    SR_CU(cudaMalloc(&dPnV,szC));SR_CU(cudaMalloc(&dPzV,szC));SR_CU(cudaMalloc(&dPpV,szC));
    SR_CU(cudaMalloc(&dEvV,szC));
    SR_CU(cudaMemcpy(dCnV,hCn,szC,cudaMemcpyHostToDevice));
    SR_CU(cudaMemcpy(dCzV,hCz,szC,cudaMemcpyHostToDevice));
    SR_CU(cudaMemcpy(dCpV,hCp,szC,cudaMemcpyHostToDevice));
    _Sr_GpuQtsNormalizeK<<<(nel+255)/256,256>>>(dCnV,dCzV,dCpV,dPnV,dPzV,dPpV,dEvV,nel);
    SR_CU(cudaGetLastError());SR_CU(cudaDeviceSynchronize());
    SR_CU(cudaMemcpy(hEvG,dEvV,szC,cudaMemcpyDeviceToHost));
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
      double cn=0,cz=0,cp=0;
      for(int k=0;k<K;k++){
        double x=(double)hX[m*K+k];
        cn+=x*(double)hAn[k*N+n];
        cz+=x*(double)hAz[k*N+n];
        cp+=x*(double)hAp[k*N+n];
      }
      double qn=cn*cn,qz=cz*cz,qp=cp*cp;
      double z=qn+qz+qp;
      double iz=(z>1e-12)?1.0/z:0.0;
      hRef[m*N+n]=(float)(qp*iz-qn*iz);
    }
    int errs=0; float maxErr=0.f; int shown=0;
    for(int i=0;i<nel;i++){
      float d=hEvG[i]-hRef[i]; if(d<0)d=-d;
      if(d>maxErr) maxErr=d;
      if(d>0.15f){
        errs++;
        if(shown<16){
          std::fprintf(stderr,"  ERR [%d,%d] E_gpu=%.4f E_ref=%.4f diff=%.4f\n",i/N,i%N,hEvG[i],hRef[i],d);
          shown++;
        }
      }
    }
    totalErrs+=errs;
    float* hPnH=(float*)std::malloc(szC);
    float* hPzH=(float*)std::malloc(szC);
    float* hPpH=(float*)std::malloc(szC);
    SR_CU(cudaMemcpy(hPnH,dPnV,szC,cudaMemcpyDeviceToHost));
    SR_CU(cudaMemcpy(hPzH,dPzV,szC,cudaMemcpyDeviceToHost));
    SR_CU(cudaMemcpy(hPpH,dPpV,szC,cudaMemcpyDeviceToHost));
    int normF=0,rangeF=0; float maxNE=0;
    for(int i=0;i<nel;i++){
      float sp=hPnH[i]+hPzH[i]+hPpH[i];
      float nd=sp-1.f; if(nd<0)nd=-nd;
      if(nd>maxNE)maxNE=nd;
      if(nd>0.01f)normF++;
      if(hEvG[i]<-1.01f||hEvG[i]>1.01f)rangeF++;
    }
    std::fprintf(stderr,"  pipeline: maxErr=%.6f errs=%d/%d (tol=0.15)\n",maxErr,errs,nel);
    std::fprintf(stderr,"  P normalization: maxErr=%.6f fails=%d\n",maxNE,normF);
    std::fprintf(stderr,"  E range [-1,1]: fails=%d\n",rangeF);
    int ca=0,cb=0,cc=0;
    for(int i=0;i<K*N;i++){
      if(hAn[i]<-0.01f||hAn[i]>0.01f)ca++;
      if(hAz[i]<-0.01f||hAz[i]>0.01f)cb++;
      if(hAp[i]<-0.01f||hAp[i]>0.01f)cc++;
    }
    std::fprintf(stderr,"  amplitude activity: neg=%d zero=%d pos=%d (total=%d)\n",ca,cb,cc,K*N);
    int cap=(M<=16&&N<=16)?nel:8;
    for(int i=0;i<cap;i++){
      std::fprintf(stderr,"  [%d,%d] E_gpu=%.4f E_ref=%.4f Cn=%.2f Cz=%.2f Cp=%.2f\n",
        i/N,i%N,hEvG[i],hRef[i],hCn[i],hCz[i],hCp[i]);
    }
    std::free(hX);std::free(hAn);std::free(hAz);std::free(hAp);
    std::free(hCn);std::free(hCz);std::free(hCp);
    std::free(hEvG);std::free(hRef);
    std::free(hPnH);std::free(hPzH);std::free(hPpH);
    cudaFree(dCnV);cudaFree(dCzV);cudaFree(dCpV);
    cudaFree(dPnV);cudaFree(dPzV);cudaFree(dPpV);cudaFree(dEvV);
  }
  std::fprintf(stderr,"[QTS] total errors: %d\n",totalErrs);
  return totalErrs;
}

/* Backward through Born normalize */
__global__ void _Sr_GpuQtsBackNormK(
    const float* __restrict__ dE,
    const float* __restrict__ cn, const float* __restrict__ cz, const float* __restrict__ cp,
    float* __restrict__ dcn, float* __restrict__ dcz, float* __restrict__ dcp, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float de = dE[i];
  float a = cn[i], b = cz[i], c = cp[i];
  float a2 = a*a, b2 = b*b, c2 = c*c;
  float z = a2 + b2 + c2;
  float iz2 = (z > 1e-12f) ? (1.0f / (z*z)) : 0.0f;
  dcn[i] = de * (-2.0f*a*(b2 + 2.0f*c2)) * iz2;
  dcz[i] = de * (-2.0f*b*(c2 - a2)) * iz2;
  dcp[i] = de * ( 2.0f*c*(2.0f*a2 + b2)) * iz2;
}

__global__ void _Sr_GpuQtsMseGradK(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ grad, float* __restrict__ loss_buf, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float d = pred[i] - target[i];
  grad[i] = d;
  loss_buf[i] = d * d;
}

__global__ void _Sr_GpuAddK(float* a, const float* b, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void _Sr_GpuQtsSgdK(float* w, const float* g, float lr, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) w[i] -= lr * g[i];
}

__global__ void _Sr_GpuQtsTransposeK(const float* in, float* out, int rows, int cols) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * cols) return;
  int r = i / cols, c = i % cols;
  out[c * rows + r] = in[r * cols + c];
}

__global__ void _Sr_GpuSumReduceK(const float* in, float* out, int n) {
  __shared__ float s[256];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < n; i += 256) sum += in[i];
  s[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *out = s[0];
}

__global__ void _Sr_GpuQtsTernaryToAmplK(
    const float* w, float* an, float* az, float* ap, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float v = w[i];
  an[i] = (v < -0.5f) ? 1.0f : 0.0f;
  az[i] = (v > -0.5f && v < 0.5f) ? 1.0f : 0.0f;
  ap[i] = (v > 0.5f) ? 1.0f : 0.0f;
}

__global__ void _Sr_GpuQtsRandTernaryK(float* out, int n, unsigned int seed) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  unsigned int s = seed ^ (unsigned int)(i * 2654435761u);
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  int v = (int)(s % 3u) - 1;
  out[i] = (float)v;
}

__global__ void _Sr_GpuQtsNaiveGemmK(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  if (row >= M || col >= N) return;
  float sum = 0.0f;
  for (int k = 0; k < K; k++) sum += A[row*K+k] * B[k*N+col];
  C[row*N+col] = sum;
}

extern "C" double Sr_GpuQtsTrain(int M, int K, int N, int epochs) {
  _sr_gpu_init();
  if (!_sr_gpu_ok) { std::fprintf(stderr,"GPU not available\n"); return -1.0; }
  std::fprintf(stderr,"M=%d K=%d N=%d epochs=%d\n",M,K,N,epochs);
  std::fprintf(stderr,"Task: learn ternary weights W_true via QTS backprop\n");
  size_t szW=(size_t)K*N*sizeof(float), szY=(size_t)M*N*sizeof(float);
  float *dX=nullptr, *dWtrue=nullptr, *dY=nullptr;
  SR_CU(cudaMalloc(&dX,(size_t)M*K*sizeof(float)));
  SR_CU(cudaMalloc(&dWtrue,szW)); SR_CU(cudaMalloc(&dY,szY));
  float *dAn=nullptr, *dAz=nullptr, *dAp=nullptr;
  SR_CU(cudaMalloc(&dAn,szW)); SR_CU(cudaMalloc(&dAz,szW)); SR_CU(cudaMalloc(&dAp,szW));
  float *dCn=nullptr, *dCz=nullptr, *dCp=nullptr;
  float *dPn=nullptr, *dPz=nullptr, *dPp=nullptr, *dE=nullptr;
  SR_CU(cudaMalloc(&dCn,szY)); SR_CU(cudaMalloc(&dCz,szY)); SR_CU(cudaMalloc(&dCp,szY));
  SR_CU(cudaMalloc(&dPn,szY)); SR_CU(cudaMalloc(&dPz,szY)); SR_CU(cudaMalloc(&dPp,szY));
  SR_CU(cudaMalloc(&dE,szY));
  float *dDE=nullptr, *dDCn=nullptr, *dDCz=nullptr, *dDCp=nullptr;
  SR_CU(cudaMalloc(&dDE,szY)); SR_CU(cudaMalloc(&dDCn,szY));
  SR_CU(cudaMalloc(&dDCz,szY)); SR_CU(cudaMalloc(&dDCp,szY));
  float *dGAn=nullptr, *dGAz=nullptr, *dGAp=nullptr;
  SR_CU(cudaMalloc(&dGAn,szW)); SR_CU(cudaMalloc(&dGAz,szW)); SR_CU(cudaMalloc(&dGAp,szW));
  float *dXt=nullptr;
  SR_CU(cudaMalloc(&dXt,(size_t)K*M*sizeof(float)));
  float *dLoss buffersBuf=nullptr, *dLoss buffersVal=nullptr;
  SR_CU(cudaMalloc(&dLoss buffersBuf,szY)); SR_CU(cudaMalloc(&dLoss buffersVal,sizeof(float)));
  _Sr_GpuQtsRandFillK<<<(M*K+255)/256,256>>>(dX,M*K,0xCAFE0001u);
  _Sr_GpuQtsRandTernaryK<<<(K*N+255)/256,256>>>(dWtrue,K*N,0xBEEF0001u);
  { float *dAnT=nullptr, *dAzT=nullptr, *dApT=nullptr;
    SR_CU(cudaMalloc(&dAnT,szW)); SR_CU(cudaMalloc(&dAzT,szW)); SR_CU(cudaMalloc(&dApT,szW));
    _Sr_GpuQtsTernaryToAmplK<<<(K*N+255)/256,256>>>(dWtrue,dAnT,dAzT,dApT,K*N);
    dim3 b0(16,16); dim3 g0((N+15)/16,(M+15)/16);
    _Sr_GpuQtsNaiveGemmK<<<g0,b0>>>(dX,dAnT,dCn,M,N,K);
    _Sr_GpuQtsNaiveGemmK<<<g0,b0>>>(dX,dAzT,dCz,M,N,K);
    _Sr_GpuQtsNaiveGemmK<<<g0,b0>>>(dX,dApT,dCp,M,N,K);
    _Sr_GpuQtsNormalizeK<<<(M*N+255)/256,256>>>(dCn,dCz,dCp,dPn,dPz,dPp,dY,M*N);
    cudaFree(dAnT); cudaFree(dAzT); cudaFree(dApT); }
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAn,K*N,0xFACE0001u);
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAz,K*N,0xFACE0002u);
  _Sr_GpuQtsRandFillK<<<(K*N+255)/256,256>>>(dAp,K*N,0xFACE0003u);
  _Sr_GpuQtsNormAmplK<<<(K*N+255)/256,256>>>(dAn,dAz,dAp,K*N);
  _Sr_GpuQtsTransposeK<<<(M*K+255)/256,256>>>(dX,dXt,M,K);
  SR_CU(cudaGetLastError()); SR_CU(cudaDeviceSynchronize());
  dim3 nB(16,16);
  dim3 gFwd((N+15)/16,(M+15)/16);
  dim3 gWgr((N+15)/16,(K+15)/16);
  float lr = 0.001f;
  float hostLoss buffers = 0.0f;
  int normN = M*N;
  int wN = K*N;
  std::fprintf(stderr,"Start (lr=0.001)\n");
  for (int ep = 0; ep < epochs; ep++) {
    _Sr_GpuQtsNaiveGemmK<<<gFwd,nB>>>(dX,dAn,dCn,M,N,K);
    _Sr_GpuQtsNaiveGemmK<<<gFwd,nB>>>(dX,dAz,dCz,M,N,K);
    _Sr_GpuQtsNaiveGemmK<<<gFwd,nB>>>(dX,dAp,dCp,M,N,K);
    _Sr_GpuQtsNormalizeK<<<(normN+255)/256,256>>>(dCn,dCz,dCp,dPn,dPz,dPp,dE,normN);
    _Sr_GpuQtsMseGradK<<<(normN+255)/256,256>>>(dE,dY,dDE,dLoss buffersBuf,normN);
    _Sr_GpuSumReduceK<<<1,256>>>(dLoss buffersBuf,dLoss buffersVal,normN);
    _Sr_GpuQtsBackNormK<<<(normN+255)/256,256>>>(dDE,dCn,dCz,dCp,dDCn,dDCz,dDCp,normN);
    _Sr_GpuQtsNaiveGemmK<<<gWgr,nB>>>(dXt,dDCn,dGAn,K,N,M);
    _Sr_GpuQtsNaiveGemmK<<<gWgr,nB>>>(dXt,dDCz,dGAz,K,N,M);
    _Sr_GpuQtsNaiveGemmK<<<gWgr,nB>>>(dXt,dDCp,dGAp,K,N,M);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAn,dGAn,lr,wN);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAz,dGAz,lr,wN);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAp,dGAp,lr,wN);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAn,dAn,0.001f,wN);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAz,dAz,0.001f,wN);
    _Sr_GpuQtsSgdK<<<(wN+255)/256,256>>>(dAp,dAp,0.001f,wN);
    if (ep % (epochs/10 > 0 ? epochs/10 : 1) == 0 || ep == epochs-1) {
      SR_CU(cudaDeviceSynchronize());
      SR_CU(cudaMemcpy(&hostLoss buffers,dLoss buffersVal,sizeof(float),cudaMemcpyDeviceToHost));
      hostLoss buffers /= (float)normN;
      std::fprintf(stderr,"  [ep %4d/%d] MSE=%.6f\n",ep,epochs,hostLoss buffers);
    }
    if (ep == epochs*7/10) { lr *= 0.3f; std::fprintf(stderr,"  lr= %.6f\n",lr); }
    if (ep == epochs*9/10) { lr *= 0.3f; std::fprintf(stderr,"  lr= %.6f\n",lr); }
  }
  SR_CU(cudaDeviceSynchronize());
  SR_CU(cudaMemcpy(&hostLoss buffers,dLoss buffersVal,sizeof(float),cudaMemcpyDeviceToHost));
  hostLoss buffers /= (float)normN;
  std::fprintf(stderr,"Final MSE=%.6f\n",hostLoss buffers);
  float* hAn = new float[wN]; float* hAz = new float[wN];
  float* hAp = new float[wN]; float* hWt = new float[wN];
  SR_CU(cudaMemcpy(hAn,dAn,szW,cudaMemcpyDeviceToHost));
  SR_CU(cudaMemcpy(hAz,dAz,szW,cudaMemcpyDeviceToHost));
  SR_CU(cudaMemcpy(hAp,dAp,szW,cudaMemcpyDeviceToHost));
  SR_CU(cudaMemcpy(hWt,dWtrue,szW,cudaMemcpyDeviceToHost));
  int match=0, total=wN;
  int cnt_neg=0, cnt_zero=0, cnt_pos=0;
  for(int i=0;i<total;i++) {
    float an2=hAn[i]*hAn[i], az2=hAz[i]*hAz[i], ap2=hAp[i]*hAp[i];
    int collapsed = 0;
    if (an2 >= az2 && an2 >= ap2) { collapsed = -1; cnt_neg++; }
    else if (ap2 >= az2 && ap2 >= an2) { collapsed = 1; cnt_pos++; }
    else { collapsed = 0; cnt_zero++; }
    if (collapsed == (int)hWt[i]) match++;
  }
  float accuracy = 100.0f * (float)match / (float)total;
  std::fprintf(stderr,"Collapse accuracy: %d/%d = %.1f%%\n",match,total,accuracy);
  std::fprintf(stderr,"Collapsed distribution: neg=%d zero=%d pos=%d\n",cnt_neg,cnt_zero,cnt_pos);
  std::fprintf(stderr,"First 8 weights:\n");
  for(int i=0;i<8 && i<total;i++) {
    float an2=hAn[i]*hAn[i], az2=hAz[i]*hAz[i], ap2=hAp[i]*hAp[i];
    int c = (an2>=az2&&an2>=ap2) ? -1 : (ap2>=az2&&ap2>=an2) ? 1 : 0;
    float z=an2+az2+ap2; float pn=(z>0?an2/z:0),pz=(z>0?az2/z:0),pp=(z>0?ap2/z:0);
    std::fprintf(stderr,"  w[%d] true=%+d coll=%+d | P(-1)=%.3f P(0)=%.3f P(+1)=%.3f\n",
      i,(int)hWt[i],c,pn,pz,pp);
  }
  delete[] hAn; delete[] hAz; delete[] hAp; delete[] hWt;
  cudaFree(dX); cudaFree(dWtrue); cudaFree(dY);
  cudaFree(dAn); cudaFree(dAz); cudaFree(dAp);
  cudaFree(dCn); cudaFree(dCz); cudaFree(dCp);
  cudaFree(dPn); cudaFree(dPz); cudaFree(dPp); cudaFree(dE);
  cudaFree(dDE); cudaFree(dDCn); cudaFree(dDCz); cudaFree(dDCp);
  cudaFree(dGAn); cudaFree(dGAz); cudaFree(dGAp);
  cudaFree(dXt); cudaFree(dLoss buffersBuf); cudaFree(dLoss buffersVal);
  return (double)accuracy;
}

