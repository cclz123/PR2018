function [A_dual E_dual] = LK(D)

if  nargin < 2
    lambda = 0.1;%0.01
end
[m n] = size(D);

% if nargin < 3
%     error('Too few arguments') ;
% end

if nargin < 3
    tol = 1e-15;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 4
    maxIter = 100000;
elseif maxIter == -1
    maxIter = 1000;
end

DISPLAY_EVERY = 10 ;

% initialize
Y = D;
% [U S V] = svd(D);
% Y = U * V';
norm_two = norm(Y, 2);
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;
% norm_two = norm_two / dual_norm;
% norm_inf = norm_inf / dual_norm;
obj_v = D(:)' * Y(:);

A_dual = zeros( m, n);
E_dual = zeros( m, n);
dt_dual = cell(1,n) ;
dt_dual_matrix = zeros(m, n) ;
% Jn = size(J{1}, 2) ; % the number of parameters of the transformation

mu = 0.1 ;%1.25/norm(D)
rho = 1.01;%1.05

d_norm = norm(D, 'fro');

iter = 0;
converged = false;
while ~converged       
    iter = iter + 1;
    
    temp_T = D + dt_dual_matrix - E_dual + (1/mu)*Y;
    [U S V] = svd(temp_T, 'econ');
    diagS = diag(S);
    A_dual = U * diag(pos(diagS-1/mu)) * V';
    
    temp_T = D + dt_dual_matrix - A_dual + (1/mu)*Y;
    E_dual = sign(temp_T) .* pos( abs(temp_T) - lambda/mu );
    
    temp_T = D - E_dual - A_dual + (1/mu)*Y;
    for i = 1 : n
        %dt_dual{i} =  - J{i}'*temp_T(:,i) ;
        %dt_dual_matrix(:, i) = J{i}*dt_dual{i} ;
    end
    
    Z = D + dt_dual_matrix - A_dual - E_dual;
    Y = Y + mu*Z;
    
    obj_v = D(:)'*Y(:);
    
    mu = mu*rho;
    stoppingCriterion = norm(Z, 'fro') / d_norm;
    
    if mod( iter, DISPLAY_EVERY) == 0
%         disp(['#Iteration ' num2str(iter) '  rank(A) ' num2str(rank(A_dual)) ...
%             ' ||E||_0 ' num2str(length(find(abs(E_dual)>0)))...
%             ' objvalue ' num2str(obj_v) '  Stopping Criterion ' ...
%             num2str(stoppingCriterion)]);
    end    
    
    
    if stoppingCriterion <= tol
%         disp('RASL inner loop is converged at:');
%         disp(['Iteration ' num2str(iter) '  rank(A) ' num2str(rank(A_dual)) ...
%             ' ||E||_0 ' num2str(length(find(abs(E_dual)>0)))  '  Stopping Criterion ' ...
%             num2str(stoppingCriterion)]) ;
        converged = true ;
    end
    
    if ~converged && iter >= maxIter
%         disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
