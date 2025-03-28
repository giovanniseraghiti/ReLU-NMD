% Numerical experiments on sparse data compression using ReLU decomposition. 
%
% Given a nonnegative and sparse X in R^{m x n}, we look for a rank-r matrix WH 
% such that X is approximated by max(0,WH), where W is a mxr matrix and H is a rxn matrix.
% The rank r of the decomposition is chosen such that nnz(X)>r(m+n).
%
% We compare eBCD-NMD with  TSVD and the state of the art algorithms for computing ReLU decompositions, that are
% BCD-NMD, Naive-NMD, A-NMD, 3B-NMD, and EM-NMD. 
% We consider well-known sparse dataset and we compare the results based on the compression error
%
%           ||X-max(0,Theta)||_F \ ||X||_F
%
% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2024.  

clear all
close all

% Add paths
cd('../'); 
addpath(genpath('./'));

%number of random initializations
rep=1;

%Stopping criteria tolerances
param.maxit=100000000;  %maximum number of iteration
param.tol=1e-9;         %tolerance on the relative residual
param.tolerr=-1;        %tolerance on the residual variation in 10 iterations
param.time=10;          %maximum CPU time (120 for the results in the paper)

%eBCD-NMD parameters
param.alpha=1;          %starting value of the extrapolation parameter (default=1)
param.mu=0.3;           %parameter regulating the growth of alpha (default=0.3)
param.alpha_max=4;      %upper bound on the extrapolation parameter (default=4)
param.delta_bar=0.8;    %lower bound on the residual reduction, if exceeded alpha is increased (default=0.8)
param.check=0;          %if set to 1 controls boundness of the sequence and the explicit rank of Z_alpha H^T (default=0).

%Parameters A-NMD
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%Choose one or more datasets to compress
data=string([]);
datasets=[3,4];              %choose a scalar or vector of integer in [0,10] (in the paper 1:10)
for j=1:length(datasets)
number=datasets(j);          %data set

    switch number
        case 1 
            name='MNIST';
            Y=load('mnist_all.mat');
            w1=1:5:5000;     %number of images per digit (default=1000 images for each digit)
            X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
               Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
            X=double(X);
            
        case 2
            name='FASHION_MNIST';
            Y=load('fashion_mnist.mat');
            s=randsample(10000,10000);
            X=reshape(Y.array,10000,28*28);
            X=double(X(s,:)');
            
        case 3 
            name='PHANTOM';
            X = phantom('Modified Shepp-Logan', 256);  % default=256x256 resolution
            X=X.*(X>0);
        case 4
            name='SATELLITE';
            im_file     = 'Datasets/satellite.png';
            f           = 255;
            X         = double(imread(im_file))/f; X=X.*(X>0);
        case 5
            name='TREC11';
            load trec11.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
        case 6
            name='MYCIELSKIAN';
            load mycielskian10.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
        case 7
            name='RKAT';
            load rkat7_mat5.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
        case 8
            name='ROBOT';
            load robot24c1_mat5.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
        case 9
            name='LOCK';
            load lock1074.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
            %Note: EM-NMD might fail with this data set
        case 10
            name='BEACONFD';
            load lp_beaconfd.mat
            X=max(0,Problem.A);     %make sure the matrix is nonnegative
            figure; spy(X);
     end
    data=[data;sprintf(name)];                 %save the name of the dataset
    %Define the dimensions of the problem
    [m,n]=size(X);
    
    %Chose the rank according to a fixed compression ratio
    comp_rat=0.5;
    r=floor(comp_rat*nnz(X)/(m+n));          %50% compression for the results in the paper
    spar=(m*n-nnz(X))/(m*n)*100;             %sparsity of the original data
    comp_ratio=(nnz(X)-r*(m+n))/nnz(X)*100;  %check the compression ratio
    
    for i=1:rep
        rng(i) %Fix the seed for reproducibility
    
        %Random rescaled initialization
        param.W0=randn(m,r); param.H0=randn(r,n); param.Z0=X; 
        alpha=sqrt(norm(X,"fro"));
        param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;   %rescaled initial point for 3B-ReLU-NMD
        param.Theta0=param.W0*param.H0;                                                               %rescaled initial point for Latent-ReLU-NMD
    
        %eBCD-NMD algorithm
        [Theta_ebcd,err_ebcd,it_ebcd,time_ebcd]=eBCD_NMD(X,r,param);
        comp_ebcd(j,i)=norm(X-max(0,Theta_ebcd),'fro')/norm(X,'fro');    %compression error
        t_ebcd(j,i)=time_ebcd(end);                                      %CPU time
        
        %BCD-NMD algorithm
        [Theta_bcd,err_bcd,it_bcd,time_bcd]=BCD_NMD(X,r,param);
        comp_bcd(j,i)=norm(X-max(0,Theta_bcd),'fro')/norm(X,'fro');      %compression error
        t_bcd(j,i)=time_bcd(end);                                        %CPU time
    
        %3B-NMD algorithm
        [Theta_3b,err_3b,it_3b,time_3b]=NMD_3B(X,r,param);
        comp_3b(j,i)=norm(X-max(0,Theta_3b),'fro')/norm(X,'fro');        %compression error
        t_3b(j,i)=time_3b(end);                                          %CPU time
    
         %Naive-NMD algorithm
        [Theta_nai,err_nai,it_nai,time_nai]=Naive_NMD(X,r,param);
        comp_nai(j,i)=norm(X-max(0,Theta_nai),'fro')/norm(X,'fro');      %compression error
        t_nai(j,i)=time_nai(end);                                        %CPU time
    
        %A-NMD algorithm
        [Theta_anmd,err_anmd,it_anmd,time_anmd]=A_NMD(X,r,param);
        comp_anmd(j,i)=norm(X-max(0,Theta_anmd),'fro')/norm(X,'fro');    %compression error
        t_anmd(j,i)=time_anmd(end);                                      %CPU time
    
        %EM-NMD algorithm
        [Theta_em,err_em,it_em,time_em]=EM_NMD(X,r,param);
        comp_em(j,i)=norm(X-max(0,Theta_em),'fro')/norm(X,'fro');        %compression error
        t_em(j,i)=time_em(end);                                          %CPU time
    
    end
    
    %TSVD as a baseline
    [U,S,V]=svds(X,r);
    SVD_ap=U*S*V';
    SVD_comp(j,1) = norm(X-max(0,SVD_ap),'fro')/norm(X,'fro');           %compression error

end
%%

if i>1
    %Compute average compression error and time
    eBCD_NMD_t=mean(t_ebcd,2); eBCD_NMD_t_std=std(t_ebcd,0,2);
    eBCD_NMD_mean=mean(comp_ebcd,2); eBCD_NMD_std=std(comp_ebcd,0,2);
    
    BCD_NMD_t=mean(t_bcd,2); BCD_NMD_t_std=std(t_bcd,0,2); 
    BCD_NMD_mean=mean(comp_bcd,2); BCD_NMD_std=std(comp_bcd,0,2);
    
    B3_NMD_t=mean(t_3b,2);  B3_NMD_t_std=std(t_3b,0,2);
    B3_NMD_mean=mean(comp_3b,2); B3_NMD_std=std(comp_3b,0,2);
    
    Naive_NMD_t=mean(t_nai,2); Naive_NMD_t_std=std(t_nai,0,2);
    Naive_NMD_mean=mean(comp_nai,2); Naive_NMD_std=std(comp_nai,0,2);
    
    A_NMD_t=mean(t_anmd,2); A_NMD_t_std=std(t_anmd,0,2);
    A_NMD_mean=mean(comp_anmd,2); A_NMD_std=std(comp_anmd,0,2);
    
    EM_NMD_t=mean(t_em,2); EM_NMD_t_std=std(t_em,0,2);
    EM_NMD_mean=mean(comp_em,2); EM_NMD_std=std(comp_em,0,2);
else
    %Compute compression error and time without averaging
    eBCD_NMD_t=t_ebcd; eBCD_NMD_mean=comp_ebcd;
    BCD_NMD_t=t_bcd; BCD_NMD_mean=comp_bcd;
    B3_NMD_t=t_3b; B3_NMD_mean=comp_3b;
    Naive_NMD_t=t_nai; Naive_NMD_mean=comp_nai;
    A_NMD_t=t_anmd; A_NMD_mean=comp_anmd;
    EM_NMD_t=t_em; EM_NMD_mean=comp_em;
end
%%

%Display table with the relative compression error ||X-max(0,Theta)||_F \ ||X||_F
fprintf('Average relative error table\n')
G_comp=table( data, BCD_NMD_mean, eBCD_NMD_mean, B3_NMD_mean,A_NMD_mean,Naive_NMD_mean,EM_NMD_mean,SVD_comp);
disp(G_comp)
%matrices containing the mean 
mtx_avg_comp=[BCD_NMD_mean, eBCD_NMD_mean, B3_NMD_mean,A_NMD_mean,Naive_NMD_mean,EM_NMD_mean,SVD_comp];

%If one single data set with one sigle initialization is analyzed we show the time vs relative residual graph
if rep==1 && length(datasets)==1
    figure
    semilogy(time_bcd,err_bcd,'b','LineWidth',2.5); hold on
    semilogy(time_3b,err_3b,'k','LineWidth',2.5);
    semilogy(time_anmd,err_anmd,'g','LineWidth',2.5);
    semilogy(time_nai,err_nai,'Color', '#7E2F8E','LineWidth',2.5);
    semilogy(time_em,err_em,'Color', '#EDB120','LineWidth',2.5);
    semilogy(time_ebcd,err_ebcd,'r','LineWidth',2.5); 
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','EM-NMD','eBCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('CPU time (s.)','FontSize',20,'FontName','Times New Roman'); ylabel('Relative residual','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontSize',20,'FontName','Times New Roman');
end


