% Application on Euclidean distance matrix completion. 
% 
% We generate some random points p_1,...,p_m in R^n according to some distribution and we compute the matrix 
% containing pairwise squared distances, that is Theta_{ij}=||p_i-p_j||^2. 
% The rank of Theta is bounded by n+2 and therefore we set the rank of the decomposition to r=n+2.
%
% We consider the problem where we observe only the smallest entries of the  matrix (the ones below some known threshold d) 
% and we want to recover the larger distances. We model this problem using the ReLU decomposition where the observed matrix X 
% is obtained from X=max(0,dee^T-Theta), where e denotes the all oes vector of appropriate dimension. Then we define
%
%       Omega={(i,j): X_{ij} > 0 }  and  Omega^C={(i,j): X_{ij} = 0 },
% 
% and we solve
%
%       min_{Z,W,H} ||Z-(dee^T-WH)||_F^2  s.t. P_{Omega}(Z)=P_{Omega}(X) and P_{Omega^C}(Z)<=0.
%
% We show the error with the groundtruth for different percentages of observed entries.
%
% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2025.  

clear all
close all

% Add paths
cd('../'); 
addpath(genpath('./'));

%Dimensions of the problem
n=200; m=n;             %number of randomly generated points
dim=3;                  %points in R^{dim}        
r=dim+2;                %known estimate of the rank of the euclidean distance matrix
param.observation=2;    %set to 1 to observe largest entries or to 2 to observe smallest entries (default=2)


%Stopping criteria tolerances
param.maxit=100000;     %maximum number of iteration
param.tol=1e-9;         %tolerance on the relative residual
param.tolerr=-1;        %tolerance on the residual variation in 10 iterations
param.time=10;          %maximum CPU time (60 for the results in the paper)

%eBCD-NMD parameters
param.alpha=1;          %starting value of the extrapolation parameter (default=1)
param.mu=0.3;           %parameter regulating the growth of alpha (default=0.3)
param.alpha_max=4;      %upper bound on the extrapolation parameter (default=4)
param.delta_bar=0.8;    %lower bound on the residual reduction, if exceeded alpha is increased (default=0.8)
param.check=0;          %if set to 1 controls boundness of the sequence and the explicit rank of Z_alpha H^T (default=0).

%Parameters A-NMD
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%Choose the level of sparsity of the solution between o and 1 (just one scalar is accepted)
%delta=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];  %choice made in the paper 
delta=[0.2,0.5,0.8];                           %smaller example to reduce running time

%Choose the number of randomly initialized examples
rep=1;

%Allocate variables
mtx_edmc_ground=[]; std_edmc_ground=[]; mtx_edmc_obs=[]; std_edmc_obs=[];

%Distribution of the points choose between random (dist=1), spiral (dist=2) or clustered (dist=3) or add other distributions.
dist=1;

%If set to 1 it creates one figure of the distributio of the points for each randomly generated problem.
picture=0;

for i=1:length(delta)              %cycle on each percentage of observed entries
    gamma=delta(i);                %percentage of observed entries  
    if param.observation==1
        n1=floor(n*m*gamma);       %number of largest observed entries
    elseif param.observation==2
        n1=floor(n*m*(1-gamma));   %number of smallest observed entries
    end
    for j=1:rep                    %cycle on the number of random initializations
        rng(j)                     %fix the seed for reproducibility
        switch dist 
            case 1
                q=10;             %interval [0,q]^dim where to generate the random points
                S=random_points(n,dim,q,picture);
            case 2
                spir.turns=5;                %number of complete turns
                spir.radius_max=5;           %maximum radius of the spiral
                spir.height_max=10;          %maximum height of the spiral
                p=0.9;                       %perturbation
                S=spiral(n,p,spir,picture);  %made to generate points in a three dimensional space
            case 3
                %check the function to change the clusters distribution
                S=cluster(picture);          %made to generate points in a three dimensional space
        end

        %Compute Euclidean distance matrix
        D = pdist(S', 'euclidean').^2;                   %pdist computes the pairwise distances
        Theta_low = squareform(D);                       %squareform converts the distance vector to a square matrix

        %Compute the threshold d
        vec=Theta_low(:);                                %full vector form of the distance matrix
        [d1,ind1]=sort(vec,"descend"); param.d=d1(n1);   %identify d such that the percentage of observed entries is matched

        %Remove distances which are smaller than d (param.observation=1) or larger than d (param.observation=2)
        if param.observation==1
            Theta_real=Theta_low-param.d;                %Make all the values smaller than d negative
        elseif param.observation==2
            Theta_real=param.d-Theta_low;                %Make all the values larger than d negative
        end
        X=max(0,Theta_real);                             %final observed matrix

        %Random rescaled initialization
        param.W0=rand(m,r); param.H0=rand(r,n); param.Z0=X; 
        alpha=(norm(X,"fro"))^(1/2);
        param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;    %rescaled initial point for 3B-ReLU-NMD
        param.Theta0=param.W0*param.H0;                                                                %rescaled initial point for Latent-ReLU-NMD
    
        %eBCD algorithm
        [Theta_ebcd,err_ebcd,it_ebcd,time_ebcd]=eBCD_NMD_mc(X,r,param);
        ground_ebcd(j,i)=norm(Theta_ebcd-Theta_low,'fro')/norm(Theta_low,'fro');     %groundtruth error
        
        %BCD algorithm
        [Theta_bcd,err_bcd,it_bcd,time_bcd]=BCD_NMD_mc(X,r,param);
        ground_bcd(j,i)=norm(Theta_bcd-Theta_low,'fro')/norm(Theta_low,'fro');       %groundtruth error
        
        %3B-NMD method
        [Theta_3b,err_3b,it_3b,time_3b]=NMD_3B_mc(X,r,param);
        ground_3b(j,i)=norm(Theta_3b-Theta_low,'fro')/norm(Theta_low,'fro');         %groundtruth error 
        
        %Naive-NMD method
        [Theta_nai,err_nai,it_nai,time_nai]=Naive_NMD_mc(X,r,param);
        ground_nai(j,i)=norm(Theta_nai-Theta_low,'fro')/norm(Theta_low,'fro');       %groundtruth error
        
        %A-NMD method
        [Theta_anmd,err_anmd,it_anmd,time_anmd]=A_NMD_mc(X,r,param);
        ground_anmd(j,i)=norm(Theta_anmd-Theta_low,'fro')/norm(Theta_low,'fro');     %groundtruth error
        
        %BCD method without rank-1 translation
        param.W0=rand(m,r+1); param.H0=rand(r+1,n); param.Z0=X; 
        alpha=(norm(X,"fro"))^(1/2);
        param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;  %new initial point
        [Theta_bcd1,err_bcd1,it_bcd1,time_bcd1]=BCD_NMD(X,r+1,param);                                %run BCD-NMD of rank-(r+1) without translation
        ground_bcd1(j,i)=norm(Theta_bcd1-Theta_real,'fro')/norm(Theta_real,'fro');                   %groundtruth error with the translated low-rank matrix
    end


end

if rep>1
    %Compute mean and standard deviation of the groundtruth error between all random initialized experiments
    groundtruth_bcd=mean(ground_bcd)';       groundtruth_bcd_std=std(ground_bcd); 
    groundtruth_ebcd=mean(ground_ebcd)';     groundtruth_ebcd_std=std(ground_ebcd); 
    groundtruth_3b=mean(ground_3b)';         groundtruth_3b_std=std(ground_3b); 
    groundtruth_nai=mean(ground_nai)';       groundtruth_nai_std=std(ground_nai); 
    groundtruth_anmd=mean(ground_anmd)';     groundtruth_anmd_std=std(ground_anmd);
    groundtruth_bcd1=mean(ground_bcd1)';     groundtruth_bcd1_std=std(ground_bcd1); 
else
    groundtruth_bcd=ground_bcd';        
    groundtruth_ebcd=ground_ebcd';     
    groundtruth_3b=ground_3b';         
    groundtruth_nai=ground_nai';       
    groundtruth_anmd=ground_anmd';     
    groundtruth_bcd1=ground_bcd1';     
end 

%Display graphs and tables
if length(delta)>1
    %Display graph of percentage of observed entries vs relative groundtruth error
    figure
    y=linspace(0.1,delta(end),length(delta)); 
    semilogy(y,groundtruth_bcd,'-b^','LineWidth',2.5); hold on
    semilogy(y,groundtruth_3b,'--ko','LineWidth',2.5);
    semilogy(y,groundtruth_anmd,'-g*','LineWidth',2.5);
    semilogy(y,groundtruth_nai,'--o','Color', '#7E2F8E','LineWidth',2.5);
    semilogy(y,groundtruth_ebcd,'--rs','LineWidth',2.5); 
    semilogy(y,groundtruth_bcd1,'--ms','LineWidth',2.5); 
    %legend, ticks and labels
    legend({'BCD-NMD','3B-NMD','A-NMD','Naive-NMD','eBCD-NMD','(r+1)BCD-NMD'},'FontSize',20,'FontName','Times New Roman')
    xlabel('Observed entries in %','FontSize',20,'FontName','Times New Roman'); ylabel('Groudtruth error','FontSize',20,'FontName','Times New Roman')
    xticks(y); grid on
    xticklabels(arrayfun(@(v) num2str(v), delta*100, 'UniformOutput', false));  %assign to the x label the entries of vec (rank values)
    set(gca,'FontSize',20,'FontName','Times New Roman');
end


%Display table with the average groundtruth error
fprintf('Groundtruth error\n')
observed_entries=delta'*100;
G_comp=table( observed_entries,groundtruth_bcd,groundtruth_ebcd,groundtruth_3b,groundtruth_nai,groundtruth_anmd,groundtruth_bcd1);
disp(G_comp)
