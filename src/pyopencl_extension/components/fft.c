float2* CPU_FFT(int N, int R, float2* data0, float2* data1) {
    for( int Ns=1; Ns<N; Ns*=R ) {
        for( int j=0; j<N/R; j++ )
            FftIteration( j, N, R, Ns, data0, data1 );
        swap( data0, data1 );
        }
    return data0;
}

void GPU_FFT(int N, int R, int Ns, float2* dataI, float2* dataO) {
    int j = b*N + t;
    FftIteration( j, N, R, Ns, dataI, dataO );
}

void FftIteration(int j, int N, int R, int Ns, float2* data0, float2*data1){
    float2 v[R];
    int idxS = j;
    float angle = -2*M_PI*(j%Ns)/(Ns*R);
    for( int r=0; r<R; r++ ) {
        v[r] = data0[idxS+r*N/R];
        v[r] *= (cos(r*angle), sin(r*angle));
    }
    FFT<R>( v );
    int idxD = expand(j,Ns,R);
    for( int r=0; r<R; r++ )
        data1[idxD+r*Ns] = v[r];
}

void FFT<2>( float2* v ) {
    float2 v0 = v[0];
    v[0] = v0 + v[1];
    v[1] = v0 - v[1];
}

int expand(int idxL, int N1, int N2 ){
    return (idxL/N1)*N1*N2 + (idxL%N1);
}