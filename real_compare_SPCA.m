clear;
close all;
addpath(genpath(pwd))
addpath ./manopt;

mkdir('log');
dfile=['log/real_spca_', datestr(now,'yyyy-mm-dd-HHMMSS'),'.log'];
diary(dfile);
diary on;

%%%%%%%%%%%%%
RunRep('Arabidopsis.mat', 10, 0.5, 1, 25)
RunRep('Arabidopsis.mat', 10, 0.25, 1, 25)
RunRep('Arabidopsis.mat', 15, 0.5, 1, 25)
RunRep('Arabidopsis.mat', 15, 0.25, 1, 25)
%%%%%%%%%%%%%
RunRep('Leukemia.mat', 10, 0.5, 1, 25)
RunRep('Leukemia.mat', 10, 0.25, 1, 25)
RunRep('Leukemia.mat', 15, 0.5, 1, 25)
RunRep('Leukemia.mat', 15, 0.25, 1, 25)
%%%%%%%%%%%%%%
RunRep('realEQTL.small.mat', 10, 0.5, 1, 25)
RunRep('realEQTL.small.mat', 10, 0.25, 1, 25)
RunRep('realEQTL.small.mat', 15, 0.5, 1, 25)
RunRep('realEQTL.small.mat', 15, 0.25, 1, 25)
%%%%%%%%%%%%%%
RunRep('Staunton100.mat', 10, 0.5, 1, 25)
RunRep('Staunton100.mat', 10, 0.25, 1, 25)
RunRep('Staunton100.mat', 15, 0.5, 1, 25)
RunRep('Staunton100.mat', 15, 0.25, 1, 25)
%%%%%%%%%%%%%%
RunRep('Staunton200.mat', 10, 0.5, 1, 25)
RunRep('Staunton200.mat', 10, 0.25, 1, 25)
RunRep('Staunton200.mat', 15, 0.5, 1, 25)
RunRep('Staunton200.mat', 15, 0.25, 1, 25)


diary off;
system('shutdown -s');

function RunRep(dataname, r, lambda, start, reps)

    T0 = []; T1 = []; T2 = []; T3 = []; T4 = []; T5 = []; T6 = []; T7 = [];
    S0 = []; S1 = []; S2 = []; S3 = []; S4 = []; S5 = []; S6 = []; S7 = [];
    F0 = []; F1 = []; F2 = []; F3 = []; F4 = []; F5 = []; F6 = []; F7 = [];

    for num = start:reps

%         fprintf(1, '\n   --------- #%d/%d, n = %d, r = %d, mu = %.2f ---------\n', num, reps, n, r, lambda);
        fprintf(1,['\n ---------',dataname,', r = %d, mu = %.2f ---------\n'], r, lambda)

        try
            rng(17);
            A = load(dataname).X;
            m = size(A,1);
            n = size(A,2);
            AtA = A'*A;
    
            [phi_init,~] = svd(randn(n,r),0);  % random intialization
            % Rsub parameters
            option_Rsub.F_mialm = -1e10;
            option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e1; option_Rsub.tol = 5e-2;
            option_Rsub.r = r; option_Rsub.n = n; option_Rsub.mu = lambda; option_Rsub.type = 1;
    
            % Use the point returned by the Rsub as the real init point
            [phi_init] = Re_sub_spca(AtA, option_Rsub);
            Xinitial.main = phi_init;
    
            %% AMANPG
            [U, S, V] = svds(A,n);
            PCAV = V(:, 1:r);
            initx = PCAV;
            tmp = A * PCAV; maxvar = sum(tmp(:) .* tmp(:));
            D = diag(S(1:r, 1:r));
            Dsq = D.^2;
            
            Xinitial.main = phi_init;
        %     L = norm(Dsq(:), 'fro')^2*2; % esitmation of the Lipschitz constant. It is important for performance!!!
            Lu = max(Dsq) * 2;
            tol = 5e-7;
            maxiter = 30000;
            F_tol = -inf;%fv0;%out_mialm.obj;%-Inf;
    
            [X_amanpg, maxit_att_amanpg, time_amanpg, F_amanpg, ~, sparsity4, avar4, time_arr_amanpg,obj_arr_amanpg,gx_arr_amanpg] = AManPG_SPCA(Xinitial, A, lambda, Lu, tol, maxiter, F_tol);
            T6 = [T6; time_amanpg];
            F6 = [F6; F_amanpg];
            S6 = [S6; sparsity4 * 100];
    
    
            option_tr.mu = lambda;
            option_tr.n = n; option_tr.r = r;
            option_tr.tau = 0.9;
            option_tr.sigma_factor = 1.25;
            if n >= 3000
                option_tr.maxtr_iter = 90;
            elseif n >= 2000
                option_tr.maxtr_iter = 70;
            else
                option_tr.maxtr_iter = 50;
            end
            option_tr.retraction = 'retr';
            option_tr.verbose = 0;
            option_tr.x_init = Xinitial.main;
            tr_solver = SPCARTR();
            tr_solver = tr_solver.init(A, option_tr);
    
            tic;
            tr_solver = tr_solver.run(tol);
            time0 = toc;
    
            record0 = tr_solver.record;
            xopt0 = tr_solver.X; fv0 = record0.loss(end); sparsity0 = record0.sparse(end); 
            optgradnorm = tr_solver.record.gX(end);
    
            T0 = [T0; time0];
            F0 = [F0; fv0];
            S0 = [S0; sparsity0 * 100];
    
            % ================= LSq-I ==================
            option_almssn.mu = lambda;
            option_almssn.n = n; option_almssn.r = r;
            option_almssn.tau = 0.99;
            option_almssn.sigma_factor = 1.25;
            option_almssn.maxinner_iter = 300;
            option_almssn.maxnewton_iter = 10;
            option_almssn.maxcg_iter = 300;
            option_almssn.retraction = 'retr';
            option_almssn.algname = 'LSq-I';
            option_almssn.gradnorm_decay = 0.95; 
            option_almssn.gradnorm_min = 1.0e-13;
            option_almssn.verbose = 0;
            option_almssn.LS = 1;
            option_almssn.x_init = Xinitial.main;
            almssn_solver = SPCANewtonNew();
            almssn_solver = almssn_solver.init(A, option_almssn);
    
            tic;
            almssn_solver = almssn_solver.run(tol);
            time1 = toc;
    
            record1 = almssn_solver.record;
            xopt1 = almssn_solver.X; fv1 = record1.loss(end); sparsity1 = record1.sparse(end);
    
            % ================= LSq-II ==================
            option_almssn.LS = 2;
            option_almssn.x_init = Xinitial.main;
            option_almssn.algname = 'LSq-II';
            almssn_solver = SPCANewtonNew();
            almssn_solver = almssn_solver.init(A, option_almssn);
    
            tic;
            almssn_solver = almssn_solver.run(tol);
            time2 = toc;
    
            record2 = almssn_solver.record;
            xopt2 = almssn_solver.X; fv2 = record2.loss(end); sparsity2 = record2.sparse(end); 
    
    
            T1 = [T1; time1];
            F1 = [F1; fv1];
            S1 = [S1; sparsity1 * 100];
    
            T2 = [T2; time2];
            F2 = [F2; fv2];
            S2 = [S2; sparsity2 * 100];
    
            % mialm
            type = 1;
    %         option_Rsub.F_mialm = -1e10;
    %         option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e2;  option_Rsub.tol = 5e-3;
    %         option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
    %         
    %         H = A;
    %         tic;
    %         [phi_init]= Re_sub_spca(H,option_Rsub);
    %         initial_time = toc;
            
            Op = struct();
            Op.applyA = @(X) X;
            Op.applyAT = @(y) y;
    
            f = struct();
            f.cost_grad = @pca_cost_grad;
            f.data = {AtA};
            f.B = A;
            
            h = struct();
            h.cost = @(X,lambda) lambda*sum(sum(abs(X)));
            h.prox = @(X,nu,lambda) max(abs(X) - nu*lambda,0).* sign(X);
            h.data = {lambda};
    %         
    %         
    %         manifold = stiefelfactory(n,r);
    %         
    %         
    %         % options_mialm.alpha = 1/(2*abs(eigs(full(AtA),1)));
    %         
    %         options_mialm.verbosity = 0;
    %         options_mialm.max_iter = 30000; options_mialm.tol = 5e-8;
    %         options_mialm.rho = 1.05; options_mialm.tau = 0.8;
    %         options_mialm.nu0 = svds(AtA, 1)^1 * 2;
    %         options_mialm.gtol0 = 1e-0; options_mialm.gtol_decrease = 0.9;
    %         options_mialm.X0 = phi_init;
    %         options_mialm.maxitersub = 10; options_mialm.extra_iter = 10;
    %         options_mialm.verbosity = 0;
    %         options_mialm.sub_solver = 3;
    %         
    %         [X_mialm,Z_mialm,out_mialm] = mialm_SPCA(Op, manifold, f, h, options_mialm);
    % 
    %         T3 = [T3; out_mialm.time];
    %         F3 = [F3; out_mialm.obj];
    %         S3 = [S3; out_mialm.sparsity * 100];
    % 
    %         %% manpg_adap
    %         option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
    %         option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
    %         option_manpg.L = m; option_manpg.F_mialm = -inf;
    %         option_manpg.inner_iter = 100;
    %         option_manpg.global_tol = 5e-5;
    % 
    %         [X_manpg_BB, F_manpg_BB,sparsity_manpg_BB,time_manpg_BB,...
    %                     maxit_att_manpg_BB,succ_flag_manpg_BB,lins_adap_manpg,in_av_adap_manpg,record_manpg_BB]= manpg_CMS_adap(H,option_manpg,dx,V);
    % 
    %         T4 = [T4; time_manpg_BB];
    %         F4 = [F4; F_manpg_BB];
    %         S4 = [S4; sparsity_manpg_BB * 100];
    % 
            %% soc parameter
            option_soc.phi_init = phi_init; option_soc.maxiter = 30000;  option_soc.tol = 5e-8; %option_soc.dual_gap = 5e-7;
            option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
            option_soc.L= m;  option_soc.type = type;
            option_soc.F_mialm = -inf;
            option_soc.X_mialm = xopt0;
            
            [X_Soc, F_soc,sparsity_soc,time_soc,...
                soc_error_XPQ,maxit_att_soc,succ_flag_SOC]= soc_spca(AtA,option_soc);
    
            T5 = [T5; time_soc];
            F5 = [F5; F_soc];
            S5 = [S5; sparsity_soc * 100];
    
    
            % ARPG
            [U, S, V] = svd(A, 'econ');
            PCAV = V(:, 1:r);
            initx = PCAV;
            tmp = A * PCAV; maxvar = sum(tmp(:) .* tmp(:));
            D = diag(S(1:r, 1:r));
            Dsq = D.^2;
            
            Xinitial.main = phi_init;
            %     L = norm(Dsq(:), 'fro')^2*2; % esitmation of the Lipschitz constant. It is important for performance!!!
            Lu = max(Dsq) * 2;
            Ll = Lu * 0.8;
            tol = 5e-8;
            maxiter = 30000;
    %         RPMWHmaxiter = 20;
    %         RPMWHtol = 1e-3;
            RPMWHmaxiter = 50;
            RPMWHtol = 3e-3;
            F_tol = -inf;%fv0;%out_mialm.obj;%-inf;
            
            [X_arpg, maxit_att_arpg, time_arpg, F_arpg, nD5, sparsity5, avar5,time_arr_arpg,obj_arr_arpg,gx_arr_arpg] = ARPG_SPCA_real(Xinitial, A, lambda, Ll, Lu, tol, maxiter,  F_tol, RPMWHmaxiter, RPMWHtol);
            T7 = [T7; time_arpg];
            F7 = [F7; F_arpg];
            S7 = [S7; sparsity5 * 100];

        catch
            continue
        end
    end

    fprintf(1, '=========== Summary: n = %d, r = %d, mu = %.3f ==========\n', n, r, lambda);
    fprintf(1, 'ALMTR:    time = %.3fs,  sparsity = %.2f,  loss = %.2f\n',       mean(T0(1:20)), mean(S0(1:20)), mean(F0(1:20)));
    fprintf(1, 'LS-I:    time = %.3fs,  sparsity = %.2f,  loss = %.2f\n',       mean(T1(1:20)), mean(S1(1:20)), mean(F1(1:20)));
    fprintf(1, 'LS-II:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n', mean(T2(1:20)), mean(S2(1:20)), mean(F2(1:20)));
%     fprintf(1, 'MIALM:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n', mean(T3(1:20)), mean(S3(1:20)), mean(F3(1:20)));
%     fprintf(1, 'ManPG:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n', mean(T4), mean(S4), mean(F4));
    fprintf(1, 'SOC:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n', mean(T5(1:20)), mean(S5(1:20)), mean(F5(1:20)));
    fprintf(1, 'AManPG:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n', mean(T6(1:20)), mean(S6(1:20)), mean(F6(1:20)));
    fprintf(1, 'ARPG:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n\n\n\n', mean(T7(1:20)), mean(S7(1:20)), mean(F7(1:20)));
end

function [f,g] = pca_cost_grad(X,AtA)
        BX = AtA*X;
        f = -sum(sum(BX.*X));
        g = -2*BX;
end
%    [time3, time4, time6]

