function [A_hat E_hat iter] = inexact_alm_rpca(D,ww, lambda, tol, maxIter)
addpath PROPACK;

[m n] = size(D);
if nargin < 2
    ww=1;
end
if nargin < 3
    lambda = 1 / sqrt(m);
end

if nargin < 4
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 5
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros( m, n);
E_hat = zeros( m, n);
mu = 1./norm_two ;% this one can be tuned
mu_bar = mu * 1e7;
rho = 1.05;      % this one can be tuned1.25
d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;
while ~converged       
    iter = iter + 1;
    
    temp_T = D - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - ww*lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);

%     if choosvd(n, sv) == 1
%         [U S V] = lansvd(D - E_hat + (1/mu)*Y, sv, 'L');
%     else
        [U S V] = svd(D - E_hat + (1/mu)*Y, 'econ');
%     end
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu*1) * V(:, 1:svp)';    

    total_svd = total_svd + 1;
    
    Z = D - A_hat - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
%     if mod( total_svd, 10) == 0
%         disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
%             ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
%             ' stopCriterion ' num2str(stopCriterion)]);
%     end    
%     
    if ~converged && iter >= maxIter
%         disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
