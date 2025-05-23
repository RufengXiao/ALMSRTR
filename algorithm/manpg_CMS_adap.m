function [X_manpg, F_manpg,sparsity,time_manpg,iter_adap,flag_succ,num_linesearch,mean_ssn, record] =manpg_CMS_adap(H,option,d_l,V)
%min -Tr(X'*H*X)+ mu*norm(X,1) s.t. X'*X=Ir.
%%
%parameters
tic;
r = option.r;%number of col
n = option.n;%dim
mu = option.mu;
maxiter =option.maxiter+1;
tol = option.tol;
%inner_tol =  option.inner_tol;
inner_iter  =  option.inner_iter;
h=@(X) sum(mu.*sum(abs(X)));
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
t_min=1e-8; % minimum stepsize
%%
%initial point
X = option.phi_init;
%  L=2*abs(eigs(full(B),1));
%  L=2*abs(eigs(B,1));
L = 8/d_l^2.*(sin(pi/4))^2 + V;
%dx = option.L/n;
%LAM_A =  (cos(2*pi*[0:n-1]'/n)-1)/dx^2;
%HX = real(fft(bsxfun(@times,ifft( X ),LAM_A)));
HX =  H*X;
record.gX = [];
record.time = [];
record.loss = [];

F(1) = -sum(sum(X.*(HX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
alpha = 1;
t0 = 1/L; 
t =t0;
linesearch_flag = 0;
num_inexact = 0;
for iter = 2:maxiter
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = HX; % negative gradient 2AX
    %xgx = X'*ngx;
    %pgx = ngx - 0.5*X*(xgx+xgx');
    pgx= ngx; % grad or projected gradient both okay
    %% subproblem
    if alpha < t_min || num_inexact > 10
        inner_tol = max(1e-15, min(1e-13,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    if iter == 2
        [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag]=Semi_newton_matrix(n,r,X,t,X+2*t*pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
        
    else
        [ PY,num_inner(iter),Lam,  opt_sub(iter) ,in_flag]=Semi_newton_matrix(n,r,X,t,X+2*t*pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
        
    end
    
    %     if iter ==2
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] =SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n), eye(n),X, X,zeros(r),zeros(n,r), eye(r), mu*t, 50, 1e-4) ;
    %     else
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] = SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n),eye(n),X, X, dual_y, dual_z, eye(r), mu*t, 50, 1e-3) ;
    %     end
    if in_flag == 1   % subprolem total iteration.
        inner_flag = 1 + inner_flag;
    end
    alpha=1;
    D = PY-X; %descent direction D
    
    [U, SIGMA, S] = svd(PY'*PY);
    SIGMA =diag(SIGMA);
    Z = PY*(U*diag(sqrt(1./SIGMA))*S');
    % [Z,R]=qr(PY,0);       Z = Z*diag(sign(diag(R))); %old version need consider the sign
    
    HZ = H*Z;
    % HZ = real(fft(bsxfun(@times,ifft( Z ),LAM_A)));
    % AZ = real(ifft( LAM_manpg.*fft(Z) ));
    
    F_trial = -sum(sum(Z.*(HZ)))+h(Z);
    normDsquared = norm(D,'fro')^2;
    
%     if  normDsquared/t^2 < tol
%         % if  abs(F(iter)-F(iter-1))/(abs(F(iter))+1)<tol
%         break;
%     end
    %% linesearch
    while F_trial>= F(iter-1)-0.5/t*alpha*normDsquared
        alpha = 0.5*alpha;
        linesearch_flag = 1;
        num_linesearch = num_linesearch+1;
        if alpha<t_min
            num_inexact = num_inexact + 1;
            break;
        end
        PY =X + alpha*D;
        %  [U,~,V]=svd(PY,0);  Z=U*V';
        %  [Z,R]=qr(PY,0);   Z = Z*diag(sign(diag(R)));  %old version need consider the sign
        [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);
        Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        HZ= H*Z;
        %HZ = real(fft(bsxfun(@times,ifft( Z ),LAM_A)));
        F_trial =  -sum(sum(Z.*(HZ))) + h(Z);
    end
    X = Z;
    HX = HZ;
    F(iter) = F_trial;
	gX = max(max(abs(D))) / (sqrt(sum(sum(X.^2))) + 1) / t;
	sparsity = sum(sum(abs(X) < 1.0e-6)) / (n * r);
	record.gX = [record.gX gX];
	record.loss = [record.loss F_trial];
	record.time = [record.time toc];
	if toc > 120
		break;
	end
%    if F_trial < option.F_manpg
%        break;
%    end
    if  gX < option.global_tol % || F_trial < option.F_manpg + 1.0e-8
        break;
    end
    if linesearch_flag == 0
        t = t*1.01;
    else
        t = max(t0,t/1.01);
    end
    
    linesearch_flag = 0;
end
X((abs(X)<=1e-6))=0;
X_manpg = X;
time_manpg = toc;
mean_ssn = sum(num_inner)/(iter-1);

flag_succ = 1;
iter_adap = iter;
sparsity= sum(sum(X_manpg==0))/(n*r);
F_manpg = F(iter);

fprintf('ManPG_adap:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
fprintf(1,print_format, iter-1,min(F), time_manpg,sparsity, inner_flag,mean_ssn  ,gX,num_linesearch);
