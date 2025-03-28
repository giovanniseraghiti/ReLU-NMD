%Test on randomly generated matrix completion with ReLU samplig, that is
%when oonly the positive entries are observed. This is an example of matrix completion
%problem with entries missing not at random. We compare TSVD, BCD-NMD, eBCD-NMD, 
%A-NMD, 3B-NMD, and EM-NMD evaluating the error with the true solution
%matrix and the running time

% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2025.  

clear all
close all

% Add paths
cd('../'); 
addpath(genpath('./'));

%Dimensions of the matrix (set m=n=1000, r=20 for the figure in the paper)
m=1000; n=1000; r=20;

%number of randomly generated problems (set to 20 for the results on the paper)
rep=2;  

%Noise 
noise=0;     %if set to 1 the generated problems are affected by noise
gamma=0.01;  %magnitude of the noise  

%Stopping criteria tolerances
param.maxit=100000;  %maximum number of iteration
%tolerance on the relative residual
if noise==0
    param.tol=1e-9;
elseif noise==1
    param.tol=1e-2;
end
param.tolerr=1e-10;  %tolerance on the residual variation in 10 iterations
param.time=10000;    %maximum CPU time

%eBCD-NMD parameters
param.alpha=1;          %starting value of the extrapolation parameter
param.mu=0.3;           %parameter regulating the growth of alpha
param.alpha_max=4;      %upper bound on the extrapolation parameter
param.delta_bar=0.8;    %lower bound on the residual reduction, if exceeded alpha is increased
param.check=0;          %if set to 1 controls boundness of the sequence and the explicit rank of Z_alpha H^T.

%Parameters A-NMD
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

 for i=1:rep
    fprintf('Random initialization %d \n',i)
    rng(i)
    %Generate the target matrix X=max(0,WH)
    W=randn(m,r); H=randn(r,n); Theta_real=W*H;
    if noise==1   
        N=randn(m,n); N=gamma*N/norm(N,'fro')*norm(W*H,'fro');  %generate the noise matrix
        X=max(0,Theta_real+N);                                  %noisy matrix
    else
        X=max(0,Theta_real);                                    %matrix without noise
    end
    
    %Initialize the algorithm
    rng(i)
    param.W0=randn(m,r); param.H0=randn(r,n); param.Z0=X; 
    alpha=sqrt(norm(X,"fro"));
    param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;   %rescaled initial point for 3B-ReLU-NMD
    param.Theta0=param.W0*param.H0;                                                               %rescaled initial point for Latent-ReLU-NMD

    %Call to extrapolated BCD
    [Theta_ebcd,err_ebcd,it_ebcd,time_ebcd]=eBCD_NMD(X,r,param);
    cell_ebcd{i}=err_ebcd; cell_ebcd_t{i}=time_ebcd;                  %save residual and running time vectors in a cell array
    len_ebcd(i)=length(err_ebcd);  t_ebcd(i)=time_ebcd(end);          %save iteration number and global running time

    %BCD method
    [Theta_bcd,err_bcd,it_bcd,time_bcd]=BCD_NMD(X,r,param);
    cell_bcd{i}=err_bcd; cell_bcd_t{i}=time_bcd;                      %save residual and running time vectors in a cell array
    len_bcd(i)=length(err_bcd);   t_bcd(i)=time_bcd(end);             %save iteration number and global running time


    %3B-NMD method
    [Theta_3b,err_3b,it_3b,time_3b]=NMD_3B(X,r,param);
    cell_3b{i}=err_3b; cell_3b_t{i}=time_3b;                          %save residual and running time vectors in a cell array
    len_3b(i)=length(err_3b);     t_3b(i)=time_3b(end);               %save iteration number and global running time


    %Naive-NMD
    [Theta_nai,err_nai,it_nai,time_nai]=Naive_NMD(X,r,param);
    cell_nai{i}=err_nai; cell_nai_t{i}=time_nai;                      %save residual and running time vectors in a cell array
    len_nai(i)=length(err_nai); t_nai(i)=time_nai(end);               %save iteration number and global running time


    %A-NMD
    [Theta_anmd,err_anmd,it_anmd,time_anmd]=A_NMD(X,r,param);
    cell_anmd{i}=err_anmd; cell_anmd_t{i}=time_anmd;                  %save residual and running time vectors in a cell array
    len_anmd(i)=length(err_anmd); t_anmd(i)=time_anmd(end);           %save iteration number and global running time

    %EM-NMD
    [Theta_em,err_em,it_em,time_em]=EM_NMD(X,r,param);
    cell_em{i}=err_em; cell_em_t{i}=time_em;                          %save residual and running time vectors in a cell array
    len_em(i)=length(err_em); t_em(i)=time_em(end);                   %save iteration number and global running time
 end

 if rep>1
    %maximum number of iterations and maximum running time for each algorithm
    max_len_ebcd=max(len_ebcd); max_time_ebcd=max(t_ebcd);
    max_len_bcd=max(len_bcd);   max_time_bcd=max(t_bcd);
    max_len_3b=max(len_3b);     max_time_3b=max(t_3b);
    max_len_nai=max(len_nai);   max_time_nai=max(t_nai);
    max_len_anmd=max(len_anmd); max_time_anmd=max(t_anmd);
    max_len_em=max(len_em);     max_time_em=max(t_em);
    
    %Create variables to store the results
    mtx_ebcd=zeros(rep,max_len_ebcd); mtx_ebcd_t=zeros(rep,max_len_ebcd);
    mtx_bcd=zeros(rep,max_len_bcd);   mtx_bcd_t=zeros(rep,max_len_bcd);
    mtx_3b=zeros(rep,max_len_3b);     mtx_3b_t=zeros(rep,max_len_3b);
    mtx_nai=zeros(rep,max_len_nai);   mtx_nai_t=zeros(rep,max_len_nai);
    mtx_anmd=zeros(rep,max_len_anmd); mtx_anmd_t=zeros(rep,max_len_anmd);
    mtx_em=zeros(rep,max_len_em);     mtx_em_t=zeros(rep,max_len_em);
    
    %Interpolate (if needed) to have the same number of entries in every residual vector for every initialization
    for i=1:rep
        %eBCD-NMD
        mtx_ebcd(i,1:length(cell_ebcd{i}))=cell_ebcd{i}; mtx_ebcd(i,length(cell_ebcd{i})+1:end)=cell_ebcd{i}(end);
        mtx_ebcd_t(i,1:length(cell_ebcd_t{i}))=cell_ebcd_t{i}; mtx_ebcd_t(i,length(cell_ebcd_t{i})+1:end)=linspace(cell_ebcd_t{i}(end), max_time_ebcd,max_len_ebcd-length(cell_ebcd_t{i}));
        %BCD-NMD
        mtx_bcd(i,1:length(cell_bcd{i}))=cell_bcd{i}; mtx_bcd(i,length(cell_bcd{i})+1:end)=cell_bcd{i}(end);
        mtx_bcd_t(i,1:length(cell_bcd_t{i}))=cell_bcd_t{i}; mtx_bcd_t(i,length(cell_bcd_t{i})+1:end)=linspace(cell_bcd_t{i}(end), max_time_bcd,max_len_bcd-length(cell_bcd_t{i}));
        %3B-NMD
        mtx_3b(i,1:length(cell_3b{i}))=cell_3b{i}; mtx_3b(i,length(cell_3b{i})+1:end)=cell_3b{i}(end);
        mtx_3b_t(i,1:length(cell_3b_t{i}))=cell_3b_t{i}; mtx_3b_t(i,length(cell_3b_t{i})+1:end)=linspace(cell_3b_t{i}(end), max_time_3b,max_len_3b-length(cell_3b_t{i}));
        %Naive-NMD
        mtx_nai(i,1:length(cell_nai{i}))=cell_nai{i}; mtx_nai(i,length(cell_nai{i})+1:end)=cell_nai{i}(end);
        mtx_nai_t(i,1:length(cell_nai_t{i}))=cell_nai_t{i}; mtx_nai_t(i,length(cell_nai_t{i})+1:end)=linspace(cell_nai_t{i}(end), max_time_nai,max_len_nai-length(cell_nai_t{i}));
        %A-NMD
        mtx_anmd(i,1:length(cell_anmd{i}))=cell_anmd{i}; mtx_anmd(i,length(cell_anmd{i})+1:end)=cell_anmd{i}(end);
        mtx_anmd_t(i,1:length(cell_anmd_t{i}))=cell_anmd_t{i}; mtx_anmd_t(i,length(cell_anmd_t{i})+1:end)=linspace(cell_anmd_t{i}(end), max_time_anmd,max_len_anmd-length(cell_anmd_t{i}));
        %EM-NMD
        mtx_em(i,1:length(cell_em{i}))=cell_em{i}; mtx_em(i,length(cell_em{i})+1:end)=cell_em{i}(end);
        mtx_em_t(i,1:length(cell_em_t{i}))=cell_em_t{i}; mtx_em_t(i,length(cell_em_t{i})+1:end)=linspace(cell_em_t{i}(end), max_time_em,max_len_em-length(cell_em_t{i}));
    end
    
    %Compute average relative residual and average time per iteration vectors
    avg_ebcd=mean(mtx_ebcd); avg_ebcd_t=mean(mtx_ebcd_t); 
    avg_bcd=mean(mtx_bcd);   avg_bcd_t=mean(mtx_bcd_t); 
    avg_3b=mean(mtx_3b);     avg_3b_t=mean(mtx_3b_t); 
    avg_nai=mean(mtx_nai);   avg_nai_t=mean(mtx_nai_t); 
    avg_anmd=mean(mtx_anmd); avg_anmd_t=mean(mtx_anmd_t); 
    avg_em=mean(mtx_em);     avg_em_t=mean(mtx_em_t); 
end

%%
%Compute average CPU time for the box plot
eBCD_NMD_t=sum(t_ebcd)/rep;  eBCD_NMD_t=eBCD_NMD_t';                             %averaged running time for eBCD-NMD
BCD_NMD_t=sum(t_bcd)/rep; BCD_NMD_t=BCD_NMD_t';                                  %averaged running time for BCD-NMD
B3_NMD_t=sum(t_3b)/rep; B3_NMD_t=B3_NMD_t';                                      %averaged running time for 3B-NMD
Naive_NMD_t=sum(t_nai)/rep; Naive_NMD_t=Naive_NMD_t';                            %averaged running time for Naive-NMD
A_NMD_t=sum(t_anmd)/rep; A_NMD_t=A_NMD_t';                                       %averaged running time for A-NMD
EM_NMD_t=sum(t_em)/rep; EM_NMD_t=EM_NMD_t';                                      %averaged running time for EM-NMD

%Compute standard deviation for the CPU time
eBCD_NMD_t_std=std(t_ebcd)';
BCD_NMD_t_std=std(t_bcd)';
B3_NMD_t_std=std(t_3b)';
Naive_NMD_t_std=std(t_nai)';
A_NMD_t_std=std(t_anmd)';
EM_NMD_t_std=std(t_em)';

if rep>1    
    %Plot average time vs average relative residual 
    figure
    loglog(avg_bcd_t,avg_bcd,'b','LineWidth',2.5); hold on
    loglog(avg_3b_t,avg_3b,'k','LineWidth',2.5);
    loglog(avg_anmd_t,avg_anmd,'-.g','LineWidth',2.5);
    loglog(avg_nai_t,avg_nai,'Color', '#7E2F8E','Linestyle','-.','LineWidth',2.5);
    loglog(avg_em_t,avg_em,'Color', '#EDB120','Linestyle','--','LineWidth',2.5);
    loglog(avg_ebcd_t,avg_ebcd,'-.r','LineWidth',2.5); 
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','EM-NMD','eBCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('CPU time (s.)','FontSize',20,'FontName','Times New Roman'); ylabel('Relative residual','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontSize',20,'FontName','Times New Roman');
    %Fix axis if needed
    if noise==1
        axis([0 4 0.8*1e-2 1]);
    else
        axis([0 37 0.8*1e-9 1]);
    end
    
    %Box plot of the running time for a fixed value of the rank  
    figure
    boxchart([t_bcd',t_ebcd',t_3b',t_nai',t_anmd',t_em'])
    set(gca,'XTickLabel',{'BCD-NMD','eBCD-NMD','3B-NMD','Naive-NMD','A-NMD','EM-NMD'},'FontSize',20,'FontName','Times New Roman');
    ylabel('Time (s.)','FontSize',20,'FontName','Times New Roman')

    %Table of the average running time
    fprintf('Average running time\n')
    G_t=table(eBCD_NMD_t,BCD_NMD_t,B3_NMD_t,Naive_NMD_t,A_NMD_t,EM_NMD_t);
    disp(G_t)

    %Table of standard deviation on the running time
    fprintf('Standard deviation on running time\n')
    G_t_std=table(eBCD_NMD_t_std,BCD_NMD_t_std,B3_NMD_t_std,Naive_NMD_t_std,A_NMD_t_std,EM_NMD_t_std);
    disp(G_t_std)
    
else
    %Plot time vs relative residual without averaging if just one example is generated
    figure
    loglog(time_bcd,err_bcd,'b','LineWidth',2.5); hold on
    loglog(time_3b,err_3b,'k','LineWidth',2.5);
    loglog(time_anmd,err_anmd,'-.g','LineWidth',2.5);
    loglog(time_nai,err_nai,'Color', '#7E2F8E','Linestyle','-.','LineWidth',2.5);
    loglog(time_em,err_em,'Color','Linestyle','--', '#EDB120','LineWidth',2.5);
    loglog(time_ebcd,err_ebcd,'-.r','LineWidth',2.5); 
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','EM-NMD','eBCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('CPU time (s.)','FontSize',20,'FontName','Times New Roman'); ylabel('Relative residual','FontSize',20,'FontName','Times New Roman')
    set(gca,'FontSize',20,'FontName','Times New Roman');
    %Fix axis if needed
    if noise==1
        axis([0 4 0.8*1e-2 1]);
    elseif noise==0
        axis([0 37 0.8*1e-9 1]);
    end

end

