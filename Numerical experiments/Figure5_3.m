% Lower-dimensional embedding of text data with the Threshold Similarity Matching (TSM).
% Given higher-dimensional inputs z_1,...,z_m in R^N (word-document frequencies) we compute the similarity matrix
%         
%         S_{ij}=max(0,<z_i,z_j>-tau*||z_i|| ||z_j||).
%
% Then, we find a rank-r ReLU decomposition of S with r<N, that is S=max(0,Theta) and we extract the
% lower-dimensional representation y_1,...,y_m in R^r from Theta. More details in 
% Saul, Lawrence K. "A geometrical connection between sparse and low-rank matrices and its application to manifold learning." (2022).
%
% This lower-dimensionality reduction technique is defined as tau-faithful  and it preserves norms and small angles (cos(z_i,z_j)>tau), 
% while large angles (cos(z_i,z_j)<tau) remain large. The error of the embedding is evaluated considering the mean angular deviation,
% which measure the difference between the small angles of the higher and lower dimensional points (compare_XY() function).
%
% In the script we compare TSVD, BCD-NMD, eBCD-NMD, A-NMD, 3B-NMD, and EM-NMD for solving the Threshold similarity matching (TSM) 
% on text data. We display the results for increasing values of the approximation rank.
% 
% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2025.  

clear all
close all
clc

% Add paths
cd('../'); 
addpath(genpath('./'));

%number of randomly generated problems (set to 1 a single run is needed)
rep=1;  

%Stopping criteria tolerances
param.maxit=100000;  %maximum number of iteration
param.tol=1e-9;      %tolerance on the relative residual
param.tolerr=-1;     %tolerance on the residual variation in 10 iterations
param.time=10;       %maximum CPU time (60 for the results in the paper)

%eBCD-NMD parameters
param.alpha=1;          %starting value of the extrapolation parameter (default=1)
param.mu=0.3;           %parameter regulating the growth of alpha (default=0.3)
param.alpha_max=4;      %upper bound on the extrapolation parameter (default=4)
param.delta_bar=0.8;    %lower bound on the residual reduction, if exceeded alpha is increased (default=0.8)
param.check=0;          %if set to 1 controls boundness of the sequence and the explicit rank of Z_alpha H^T (default=0).

%Parameters A-NMD
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

% Eigensolver options extract the embedded points from the similarity matrix in the lower-dimensional space
eig_opts.issym = 1;
eig_opts.tol = 1e-8;
eig_opts.fail = 'keep';
eig_opts.maxit = 500;

%Load data set and parameter for the Threshold Similarity Matrix (TSM) embedding
data=1;     %choose the data set (data=1 k1b, data=2 hitech)
switch data
    case 1   %k1b data set
        load('k1b.mat'); 
        tau=0.17;           %TSM parameter 
    case 2   %hitech data set
        load('hitech.mat'); 
        tau=0.08;           %TSM parameter
end
%shuffle the columns of the dataset
[a,index]=sort(classid); 
X=full(dtm); X=X(index,:)';  %words x documents matrix

%Compute similarity matrix
x=vecnorm(X);   XX=X'*X-tau*(x')*x; S=max(0,XX);   % similarity matrix of size number of documents x number of documents
[m,n]=size(S);  spar=(m*n-nnz(S))/(m*n)*100;

%Choose the values of the rank to test
switch data
    case 1   %k1b data set
        Rank=[25;50];
        %Rank=[25;50;75;100;125];    %choice in the paper
    case 2   %hitech data set
        Rank=[75;100;125;150;175];   %choice in the paper
end

for i=1:length(Rank)
    r=Rank(i);        %approximation rank
    fprintf('\n Running experimnt for rank=%d \n',r);

    for j=1:rep
        
        %fix the seed for reproducibility
        rng(j)      
        
        %Random rescaled initialization
        param.W0=randn(m,r); param.H0=randn(r,n); param.Z0=S; 
        alpha=sqrt(norm(S,"fro"));
        param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;   %rescaled initial point for 3B-ReLU-NMD
        param.Theta0=param.W0*param.H0;                                                               %rescaled initial point for Latent-ReLU-NMD
        
        %SVD embedding
        [U,S1,V]=svds(S,r);
        SVD_ap=U*S1*V';
        %Compute the error
        [mean_ang_svd(j,i),~] = compare_XY(X,SVD_ap,tau);
    
        %eBCD-NMD algorithm
        [Theta_ebcd,err_ebcd,it_ebcd,time_ebcd]=eBCD_NMD(S,r,param);
        % extract the lower-dimensional points from the similarity matrix
        u_ebcd = sqrt(max(0,diag(Theta_ebcd)));
        G_ebcd = Theta_ebcd + (tau/(1-tau))*(u_ebcd*u_ebcd');
        Gfun_ebcd = @(v) G_ebcd*v;
        [vG_ebcd,eG_ebcd] = eigs(Gfun_ebcd,n,r,'largestreal',eig_opts);
        T_ebcd = sqrt(max(0,eG_ebcd))*vG_ebcd';                     %lower-dimensional points
        %Compute error and iteration time
        [mad_ebcd(j,i),~] = compare_XY(X,T_ebcd,tau);               %mean angular deviation computation
        tim_ebcd(j,i)=time_ebcd(end)/it_ebcd;                       %average iteration time
    
        %BCD-NMD algorithm
        [Theta_bcd,err_bcd,it_bcd,time_bcd]=BCD_NMD(S,r,param);
        % extract the lower-dimensional points from the similarity matrix
        u_bcd = sqrt(max(0,diag(Theta_bcd)));
        G_bcd = Theta_ebcd + (tau/(1-tau))*(u_bcd*u_bcd');
        Gfun_bcd = @(v) G_bcd*v;
        [vG_bcd,eG_bcd] = eigs(Gfun_bcd,n,r,'largestreal',eig_opts);
        T_bcd = sqrt(max(0,eG_bcd))*vG_bcd';                         %lower-dimensional points
        %Compute error and iteration time
        [mad_bcd(j,i),~] = compare_XY(X,T_bcd,tau);                  %mean angular deviation computation
        tim_bcd(j,i)=time_bcd(end)/it_bcd;                           %average iteration time
    
        %3B-NMD algorithm
        [Theta_3b,err_3b,it_3b,time_3b]=NMD_3B(S,r,param);
        %extract the lower-dimensional points from the similarity matrix
        u_3b = sqrt(max(0,diag(Theta_3b)));
        G_3b = Theta_3b + (tau/(1-tau))*(u_3b*u_3b');
        Gfun_3b = @(v) G_3b*v;
        [vG_3b,eG_3b] = eigs(Gfun_3b,n,r,'largestreal',eig_opts);
        T_3b = sqrt(max(0,eG_3b))*vG_3b';                            %lower-dimensional points
        %Compute error and iteration time
        [mad_3b(j,i),~] = compare_XY(X,T_3b,tau);                    %mean angular deviation computation
        tim_3b(j,i)=time_3b(end)/it_3b;                              %average iteration time
    
        %Naive-NMD algorithm
        [Theta_nai,err_nai,it_nai,time_nai]=Naive_NMD(S,r,param);
        %extract the lower-dimensional points from the similarity matrix
        u_nai = sqrt(max(0,diag(Theta_nai)));
        G_nai = Theta_nai + (tau/(1-tau))*(u_nai*u_nai');
        Gfun_nai = @(v) G_nai*v;
        [vG_nai,eG_nai] = eigs(Gfun_nai,n,r,'largestreal',eig_opts);
        T_nai = sqrt(max(0,eG_nai))*vG_nai';                          %lower-dimensional points
        %Compute error and iteration time
        [mad_nai(j,i),~] = compare_XY(X,T_nai,tau);                   %mean angular deviation computation
        tim_nai(j,i)=time_nai(end)/it_nai;                            %average iteration time
    
        %A-NMD algorithm
        [Theta_anmd,err_anmd,it_anmd,time_anmd]=A_NMD(S,r,param);
        %extract the lower-dimensional points from the similarity matrix
        u_anmd = sqrt(max(0,diag(Theta_anmd)));
        G_anmd = Theta_anmd + (tau/(1-tau))*(u_anmd*u_anmd');
        Gfun_anmd = @(v) G_anmd*v;
        [vG_anmd,eG_anmd] = eigs(Gfun_anmd,n,r,'largestreal',eig_opts);
        T_anmd = sqrt(max(0,eG_anmd))*vG_anmd';                        %lower-dimensional points
        %Compute error and iteration time
        [mad_anmd(j,i),~] = compare_XY(X,T_anmd,tau);                  %mean angular deviation computation
        tim_anmd(j,i)=time_anmd(end)/it_anmd;                          %average iteration time
    
        %EM-NMD algorithm
        [Theta_em,err_em,it_em,time_em]=EM_NMD(S,r,param);
        %extract the lower-dimensional points from the similarity matrix
        u_em = sqrt(max(0,diag(Theta_em)));
        G_em = Theta_em + (tau/(1-tau))*(u_em*u_em');
        Gfun_em = @(v) G_em*v;
        [vG_em,eG_em] = eigs(Gfun_em,n,r,'largestreal',eig_opts);
        T_em = sqrt(max(0,eG_em))*vG_em';   
        %Compute error and iteration time%lower-dimensional points
        [mad_em(j,i),~] = compare_XY(X,T_em,tau);                       %mean angular deviation computation
        tim_em(j,i)=time_em(end)/it_em;                                 %average iteration time
    
    end
    
end
%%
if rep>1
    %Compute average and standard deviation of the mean angular deviation for each algorithm
    mean_ang_ebcd=mean(mad_ebcd)';  mad_ebcd_std=std(mad_ebcd);
    mean_ang_bcd=mean(mad_bcd)';    mad_bcd_std=std(mad_bcd);
    mean_ang_3b=mean(mad_3b)';      mad_3b_std=std(mad_3b);
    mean_ang_nai=mean(mad_nai)';    mad_nai_std=std(mad_nai);
    mean_ang_anmd=mean(mad_anmd)';  mad_anmd_std=std(mad_anmd);
    mean_ang_em=mean(mad_em)';      mad_em_std=std(mad_em);
    
    %Compute average and standard deviation of the time per iteration for each algorithm
    t_ebcd=mean(tim_ebcd)';  t_ebcd_std=std(tim_ebcd);
    t_bcd=mean(tim_bcd)';    t_bcd_std=std(tim_bcd);
    t_3b=mean(tim_3b)';      t_3b_std=std(tim_3b);
    t_nai=mean(tim_nai)';    t_nai_std=std(tim_nai);
    t_anmd=mean(tim_anmd)';  t_anmd_std=std(tim_anmd);
    t_em=mean(tim_em)';      t_em_std=std(tim_em);
else
    mean_ang_ebcd=mad_ebcd';  t_ebcd=tim_ebcd';
    mean_ang_bcd=mad_bcd';    t_bcd=tim_bcd';
    mean_ang_3b=mad_3b';      t_3b=tim_3b';
    mean_ang_nai=mad_nai';    t_nai=tim_nai';
    mean_ang_anmd=mad_anmd';  t_anmd=tim_anmd';
    mean_ang_em=mad_em';      t_em=tim_em';
end
%%
%Plots or tables
if length(Rank)>1
    %Display average mean angular deviation graph for different values of the rank
    figure
    y=linspace(10,Rank(end),length(Rank)); 
    semilogy(y,mean_ang_bcd,'-b^','LineWidth',2.5); hold on
    semilogy(y,mean_ang_3b,'--ko','LineWidth',2.5);
    semilogy(y,mean_ang_anmd,'-g*','LineWidth',2.5);
    semilogy(y,mean_ang_nai,'-o','Color', '#7E2F8E','LineWidth',2.5);
    semilogy(y,mean_ang_em,'-^','Color', '#EDB120','LineWidth',2.5);
    semilogy(y,mean_ang_ebcd,'--rs','LineWidth',2.5); 
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','EM-NMD','eBCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('Rank','FontSize',20,'FontName','Times New Roman'); ylabel('Mean angular deviation','FontSize',20,'FontName','Times New Roman')
    xticks(y); grid on
    xticklabels(arrayfun(@(v) num2str(v), Rank, 'UniformOutput', false));  %assign to the x label the entries of vec (rank values)
    set(gca,'FontSize',20,'FontName','Times New Roman');
        
    
    %Display average time per iteration for different values of the rank
    figure
    y=linspace(10,Rank(end),length(Rank)); 
    plot(y,t_bcd,'-b^','LineWidth',2.5); hold on
    plot(y,t_3b,'--ko','LineWidth',2.5);
    plot(y,t_anmd,'--gx','LineWidth',2.5);
    plot(y,t_nai,'-.o','Color', '#7E2F8E','LineWidth',2.5);
    plot(y,t_em,'-^','Color', '#EDB120','LineWidth',2.5);
    plot(y,t_ebcd,'--rs','LineWidth',2.5); 
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','EM-NMD','eBCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('Rank','FontSize',20,'FontName','Times New Roman'); ylabel('Time per iteration (s.)','FontSize',20,'FontName','Times New Roman')
    xticks(y); grid on
    xticklabels(arrayfun(@(v) num2str(v), Rank, 'UniformOutput', false));   %assign to the x label the entries of vec (rank values)
    set(gca,'FontSize',20,'FontName','Times New Roman');

end
    %Display table with average iteration time
    fprintf('Average itertaion time\n')
    G_t=table(Rank,t_ebcd,t_bcd,t_3b,t_nai,t_anmd,t_em);
    disp(G_t)

    %Display table with the average mean angular deviation
    fprintf('Mean angular deviation\n')
    G_comp=table( Rank,mean_ang_ebcd,mean_ang_bcd,mean_ang_3b,mean_ang_nai,mean_ang_anmd,mean_ang_em);
    disp(G_comp)

