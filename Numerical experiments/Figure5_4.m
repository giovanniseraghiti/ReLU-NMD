%This script produces visual representation of the compressed images using
%the eBCD-NMD algorithm on the MNIST, PHANTOM, and SATELLITE datasets

clear all
close all

% Add paths
cd('../'); 
addpath(genpath('./'));

rng(2025) %To reproduce exactly the results in the paper

%Stopping criteria tolerances
param.maxit=100000000;  %maximum number of iteration
param.tol=1e-9;         %tolerance on the relative residual
param.tolerr=-1;        %tolerance on the residual variation in 10 iterations
param.time=20;          %maximum CPU time (120 for the results in the paper)

%eBCD-NMD parameters
param.alpha=1;          %starting value of the extrapolation parameter (default=1)
param.mu=0.3;           %parameter regulating the growth of alpha (default=0.3)
param.alpha_max=4;      %upper bound on the extrapolation parameter (default=4)
param.delta_bar=0.8;    %lower bound on the residual reduction, if exceeded alpha is increased (default=0.8)
param.check=0;          %if set to 1 controls boundness of the sequence and the explicit rank of Z_alpha H^T (default=0).

%Parameters A-NMD
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%Choose the data set selecting one integer from 1 to 4
number=1;
switch number
    case 1     %MNIST dataset
        Y=load('mnist_all.mat');
        w1=1:5:5000; %Number of images for each digit
        X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
           Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
        X=double(X);
    case 2     %FASHION MNIST dataset
            Y=load('fashion_mnist.mat');
            s=randsample(10000,10000);
            X=reshape(Y.array,10000,28*28);
            X=double(X(s,:));
    case 3     %PHANTOM
        X = phantom('Modified Shepp-Logan', 256);  % 256x256 resolution
        X=X.*(X>0);
    case 4     %Satellite
        im_file     = 'Datasets/satellite.png';
        f           = 255;
        X         = double(imread(im_file))/f; X=X.*(X>0);
end

%Define the dimensions of the problem
[m,n]=size(X);

%Chose the rank according to a fixed compression ratio
comp_rat=0.5;
r=floor(comp_rat*nnz(X)/(m+n)); %r=5;
spar=(m*n-nnz(X))/(m*n)*100;            %sparsity of the original data
comp_ratio=(nnz(X)-r*(m+n))/nnz(X)*100; %compression ratio

%TSVD as a baseline
[U,S,V]=svds(X,r);
SVD_ap=U*S*V';
SVD_comp = norm(X-max(0,SVD_ap),'fro')/norm(X,'fro');

%Random rescaled initialization
param.W0=randn(m,r); param.H0=randn(r,n); param.Z0=X; 
alpha=sqrt(norm(X,"fro"));
param.W0=param.W0/norm(param.W0,'fro')*alpha; param.H0=param.H0/norm(param.H0,'fro')*alpha;   %rescaled initial point for 3B-ReLU-NMD

%Call to extrapolated BCD
[Theta_ebcd,err_ebcd,it_ebcd,time_ebcd]=eBCD_NMD(X,r,param);
comp_ebcd=norm(X-max(0,Theta_ebcd),'fro')/norm(X,'fro');
t_ebcd=time_ebcd(end);

%Plot the original images and compressed images form TSVD and ReLU-NMD, that is max(0,WH)
switch number
    case 1
        rng(2025)
        s=randsample(10000,16);
        for i=1:length(s)
            Orig_t(:,i)=reshape(reshape(X(s(i),:),28,28)',784,1);
            Theta_ebcd_t(:,i)=reshape(reshape(Theta_ebcd(s(i),:),28,28)',784,1);
            SVD_ap_t(:,i)=reshape(reshape(SVD_ap(s(i),:),28,28)',784,1);
        end
        affichage(Orig_t,4,28,28); title('Original image')
        affichage(max(0,Theta_ebcd_t),4,28,28); title('ReLU-NMD compression')
        affichage(SVD_ap_t,4,28,28); title('SVD compression')
    case 2
        rng(2025)
        s=randsample(10000,16);
        for i=1:length(s)
            Orig_t(:,i)=reshape(reshape(X(s(i),:),28,28),784,1);
            Theta_ebcd_t(:,i)=reshape(reshape(Theta_ebcd(s(i),:),28,28),784,1);
            SVD_ap_t(:,i)=reshape(reshape(SVD_ap(s(i),:),28,28),784,1);
        end
        affichage(Orig_t,4,28,28); title('Original image')
        affichage(max(0,Theta_ebcd_t),4,28,28); title('ReLU-NMD compression')
        affichage(SVD_ap_t,4,28,28); title('SVD compression')
    case 3
        figure
        imagesc(X); colormap('gray')
        title('Original image')
        figure;
        imagesc(max(0,Theta_ebcd)); colormap('gray')
        title('ReLU-NMD compression')
        figure
        imagesc(SVD_ap); colormap('gray')
        title('SVD compression')
    case 4
        figure
        imagesc(X); colormap('gray')
        title('Original image')
        figure;
        imagesc(max(0,Theta_ebcd)); colormap('gray')
        title('ReLU-NMD compression')
        figure
        imagesc(SVD_ap); colormap('gray')
        title('SVD compression')
    
end
