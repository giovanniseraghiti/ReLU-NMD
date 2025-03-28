function [Theta,err,i,time]=eBCD_NMD(X,r,param)

% Let X sparse and nonnegative, define
%
%  Omega={(i,j): X_{ij} > 0 }  and  Omega^C={(i,j): X_{ij} = 0 }
% 
% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,W,H} ||Z-WH||_F^2  s.t. P_{Omega}(Z)=P_{Omega}(X) and P_{Omega^C}(Z)<=0,
% 
% using a three blocks alternating minimization with extrapolation. Strting from (Z,W,H),
% one step of the algorithm produces a new iterate (Z(alpha),W(alpha),H(alpha)) of the form:
% 
% Indirect extrapolation step: Z_alpha=alpha*Z+(1-alpha)*W*H,
%
% Compute Q(alpha) from the economy QR decomposition of (Z_alpha)H^T,
%
% Update of W: W(alpha)=Q(alpha), 
%
% Update of H: H(alpha)=(W(alpha)^T)Z_alpha
%
% Update of Z: Z(alpha)=argmin_Z || Z - W(alpha) H(alpha) ||_F^2  s.t.  max(0,Z)=X
% 
% Restarting of the momentum parameter: if the residual decreases
%
% Accepted iterate: (W^{k+1},H^{k+1},Z^{k+1})=(W(alpha),H(alpha),Z(alpha))
%
% Otherwise:        (W^{k+1},H^{k+1},Z^{k+1})=(W^k,H^k,Z^k)
%
% Reset the extrapolation parameter alpha based on the reduction of the residual at each iteration.

%****** Input ******
%   X              : m-by-n matrix, sparse and non negative
%   r              : rank of the decomposition
%   param          : structure, containing the parameter of the model
%       .W0        = initialization of the variable W (default: randn)
%       .H0        = initialization of the variable H (default: randn)
%       .Z0        = initialization of the variable Z (default: X)
%       .alpha     = initial value for the extrapolation parameter
%       .alpha_max = upper bound for the extrapolation parameter
%       .mu        = increasing factor for the extrapolation parameter
%       .delta_bar = minimum residual ratio for step acceptance
%       .check     = if set to 1, checks the rank of (Z_alpha)H^T to detect possible rank-deficient solution,
%                    it also checks boundeness of the sequence (default=0)
%       .maxit     = maximum number of iterations (default: 1000) 
%       .tol       = tolerance on the relative error (default: 1.e-9)
%       .tolerr    = tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-10)
%       .time      = time limit (default: 60)
%       .display   = if set to 1, it diplayes error along iterations (default: 1)
% 
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of    
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)<=r.
%   err     : vector containing evolution of the relative residual along
%             iterations ||Z-WH||_F / || X ||_F
%   i       : number of iterations  
%   time    : vector containing time counter along iterations
% 
% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2025.  
tic
[m,n]=size(X); 
 if nargin < 3
    param = [];
 end
 if ~isfield(param,'W0') || ~isfield(param,'H0')
     param.W0=randn(m,r); param.H0=randn(r,n);  
 end
 if ~isfield(param,'Z0')
    param.Z0 = X; 
end
if ~isfield(param,'maxit')
    param.maxit = 1000; 
end
if ~isfield(param,'tol')
    param.tol = 1.e-9; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-10;
end
if ~isfield(param,'time')
    param.time = 60; 
end
if ~isfield(param,'alpha')
    param.alpha = 1; 
end
if ~isfield(param,'alpha_max')
    param.alpha_max = 4; 
end
if ~isfield(param,'mu')
    param.mu = 0.3; 
end
if ~isfield(param,'delta_bar')
    param.delta_bar = 0.8; 
end
if ~isfield(param,'display')
    param.display = 1;
end
if ~isfield(param,'rank')
    param.check = 0;
end

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n'; 
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2])); 
    X(X<0) = 0; 
end

%Compute fixed quantities for X
normX=norm(X,'fro');
idx=(X==0); 

%Define the initial variables
Z=param.Z0; W=param.W0; H=param.H0; Theta=W*H; S=norm(Z-W*H,'fro');

%Initialize error and time counter
err(1)=norm(Theta-Z,'fro')/normX;
time(1)=toc;

if param.display == 1
    disp('Running eBCD-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.2; alpha=param.alpha; 
for i=1:param.maxit
    tic
    %Indirect extrapolation step
    Z_alpha=alpha*Z+(1-alpha)*Theta;
    
    %Update of W computing an orthogonal basis of Z_alpha*H^T
    if param.check==1
        r1=rank(Z_alpha*H');        %Check explicitely the rank of Z_alphaH^T (always full rank in practice) 
        [W_e,~,~]=qr(Z_alpha*H',0); W_e=W_e(:,1:r1);
    else
        [W_e,~]=qr(Z_alpha*H',0);
    end

    %update of H 
    H_e=W_e'*Z_alpha; 
    
    %approzimation matrix
    Theta_e=W_e*H_e;

    if param.check==1         %check on the boundness of the entries in Omega^C that needs to remain bounded to have convergence
        if max(abs(Theta_e.*idx),[],'all')>1e10
            fprintf('The sequence might be unbounded\n') %print a warning message
        end
    end
    
    %Update of Z
    Z_e=min(0,Theta_e.*idx);
    Z_e=Z_e+X;

    %Evaluating the residual
    S_e=norm(Z_e-Theta_e,'fro'); 
    res_ratio=S_e/S;             %residual ratio
    
    %Adaptive strategy to select extrapolation parameter
    if res_ratio < 1
        W=W_e; H=H_e; Z=Z_e; S=S_e; Theta=Theta_e;  %accept the update
        if res_ratio > param.delta_bar            
            param.mu=max(param.mu,0.25*(alpha-1)); alpha=min(alpha+param.mu,param.alpha_max); %increase the extrapolation parameter if needed
            if alpha==param.alpha_max
                alpha=1;     %set the extrapolation parameter to 1 if we reach the upper bound
            end
        end
    else
        alpha=1;   %set the extrapolation parameter to 1 if the residual does not decrease
    end

    %Compute the relative residual 
    err(i+1)=S/normX;     
    
    %Stopping criteria on relative residual
    if err(i+1)<param.tol 
        time(i+1)=time(i)+toc; %needed to have same time for each iterations
         if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: ||Z-WH||/||X|| < %2.0d\n',param.tol);
        end
        break
    end
    
    %Stopping criteria if the residual is not reduced sufficiently in 10 iterations
    if i >= 11  &&  abs(err(i+1) - err(i-10)) < param.tolerr
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < %2.0d\n',param.tolerr);
        end
        break
    end
  
    %Stopping criteria on running time
    time(i+1)=time(i)+toc;
    if time(i+1)>param.time
        break
    end
    
    %Display the behavior of the residual along the iterations
    if param.display == 1 && time(i+1) >= cntdis 
        disp_time = min(60,disp_time*1.5);
        fprintf('[%2.2d : %2.2f] - ',i,100*err(i)); 
        cntdis = time(i+1)+disp_time; % display every disp_time
        numdis = numdis+1; 
        if mod(numdis,5) == 0
            fprintf('\n');
        end
    end
    
end
if param.display == 1
    fprintf('Final relative error: %2.2f%%, after %2.2d iterations. \n',100*err(i+1),i); 
end