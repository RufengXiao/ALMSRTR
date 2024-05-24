function [X_Soc, F_SOC,sparsity_soc,time_soc,error_XPQ, iter_soc, flag_succ,record]=soc_spca(B,option)
%min -Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = B'*B type = 0 or A = B  type = 1
tic;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;
if type==0 % data matrix
    A = -B'*B;
else
    A = -B;
end

h=@(X) mu*sum(sum(abs(X)));
%rho = svds(B,1)^2 + r/2;%  stepsize
%rho = svds(B,1)^2 + n*r*mu/25 + 1;
%rho = svds(B,1)^2 + n/50 ;% good for mu and r
rho = 2* svds(B,1)^1  ;%  n/30 not converge   1.9* sometimes not converge
%rho = 5*n^0.7/1.25/r;
lambda = rho;
P = option.phi_init;    Q = P;
Z = zeros(n,r); 
b=Z;  
F_ad=zeros(maxiter,1);
not_full_rank = 0;
record.gXP = [];
record.gXQ = [];
record.gX = [];
record.gQ = [];
record.gP = [];
record.time = [];
record.loss = [];
record.stops = [];
%chA = chol( 2*A + (r+lambda)*eye(d));
Ainv = inv( 2*A + (rho+lambda)*eye(n));
flag_maxiter = 0;

for itera=1:maxiter
    LZ = rho*(P-Z)+lambda*(Q-b);
    %   X=A_bar\LB;
    %  X = chA\(chA'\LZ);
    X = Ainv*LZ;
    %%%% shrinkage Q
    Q = sign(X+b).*max(0,abs(X+b)-mu/lambda);
    
    %%%% solve P
    
    Y = X + Z;
    %%%%%%%%%%%%%   svd Y'*Y
    %     [U, D, S] = svd(Y'*Y);
    %     D = diag(D);
    %     if abs(prod(D))>0
    %         P = Y*(U*diag(sqrt(1./D))*S');
    %     else
    %         not_full_rank = not_full_rank+1;
    %     end
    [U,~,V]= svd(Y,0);
    P = U*V';
    %%%%%%%%%
    Z  = Z+X-P;
    b  = b+X-Q;
    
    if type == 0 % data matrix
        AP = -(B'*(B*P));
    else
        AP = -(B*P);
    end

%     if itera>2
%normXQ = norm(X-Q,'fro');
        %normQ = norm(Q,'fro');
        %normX = norm(X,'fro');
        %normP = r;
        %normXP = norm(X-P,'fro');
        normQ = norm(Q,'fro');
        normX = norm(X,'fro');
        normP = r;
		gXQ = max(max(abs(X - Q))) / (max(normQ, normX) + 1);
		gXP = max(max(abs(X - P))) / (max(normP, normX) + 1);
		g_primal_X = max(max(abs(2*(A*X) + rho * Z + lambda * b)));
		zero_Q = (abs(Q) < 1.0e-5);
        g_primal_Q = (1 - zero_Q) .* (mu * sign(Q) - lambda * b) - zero_Q .* l1_prox_soccm(lambda * b, mu);
		PZ = P' * Z;
		g_primal_P = Z - 0.5 * P * (PZ + PZ');

		g_primal_P = max(max(abs(g_primal_P))) / (normP + 1);
		g_primal_Q = max(max(abs(g_primal_Q))) / (normQ + 1);
		g_primal_X = max(max(abs(g_primal_X))) / (normX + 1);
		record.gXP = [record.gXP gXP];
		record.gXQ = [record.gXQ gXQ];
		record.gX  = [record.gX  g_primal_X];
		record.gQ  = [record.gQ  g_primal_Q];
		record.gP  = [record.gP  g_primal_P];
		record.time  = [record.time  toc];
		stop_rule_dual = gXQ + gXP;
		stop_rule_primal = g_primal_P + g_primal_Q + g_primal_X;
        record.stops = [record.stops stop_rule_primal + stop_rule_dual];
        normXQ = norm(X-Q,'fro');
        normXP = norm(X-P,'fro');
		F_ad(itera)= sum(sum(X.*(A*P)))+h(P);
		record.loss = [record.loss F_ad(itera)];
%		if stop_rule_primal < option.tol && stop_rule_dual < option.dual_gap
        if stop_rule_primal < tol
			break;
        end
       % if  normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) <tol
       %     AP = A*P;
       %     F_ad(itera)=sum(sum(X.*(AP)))+h(P);
       %     if F_ad(itera)<=option.F_manpg + 1e-7
       %         break;
       %     end
       %      if normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) < 1e-10
       %          flag = 1; %different point
       %          break;
       %      end
       % end
        %         if   normXQ  + normXP <1e-9*r
        %             break;
        %         end
%     end

    P_old=P;
    if itera ==maxiter
        flag_maxiter =1;
    end
end
P((abs(P)<=1e-6))=0;
X_Soc=P;
time_soc= toc;
error_XPQ = norm(X-P,'fro') + norm(X-Q,'fro');
sparsity_soc= sum(sum(P==0))/(n*r);
if itera == maxiter
    flag_succ = 0; %fail
	F_SOC = F_ad(itera);
	iter_soc = itera;
    fprintf('SOC fails to converge  \n');
    
        fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
        
        print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
        fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
    else
        flag_succ = 1; % success
        F_SOC = F_ad(itera);
        iter_soc = itera;
        % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
        fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
        
        print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
        fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
end
end

function ret = l1_prox_soccm(X, lam)
	ret = sign(X) .* max(0, abs(X) - lam);
end
