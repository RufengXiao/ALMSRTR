clear;
close all;
addpath(genpath(pwd))
addpath ./manopt;

set(groot,'defaultLineLineWidth',2)

for n = [1000]
    m = 50;
    r = 20;
    lambda = 1.00;
    start = 1;
    reps = 1;
    
    for num = start:reps
        try
        seed = 1*num;
        rng(seed);
        colors = linspecer(7);
    
        A = randn(m, n);
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        
        [U, S, V] = svd(A, 'econ');
        D = diag(S(1:r, 1:r));
        D = sort(abs(randn([m 1])) .^ 4) + 1.0e-5;
    
        A = U * diag(D) * V';
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        AtA = A'*A;
    
        [phi_init,~] = svd(randn(n,r),0);  % random intialization
        Xinitial.main = phi_init;
        tol = 5e-8;
    
    
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
        tol = 5e-8;
        maxiter = 30000;
        F_tol = -inf;%fv0;%out_mialm.obj;%-Inf;
    
        [X_amanpg, maxit_att_amanpg, time_amanpg, F_amanpg, ~, sparsity4, avar4, time_arr_amanpg,obj_arr_amanpg,gx_arr_amanpg] = AManPG_SPCA(Xinitial, A, lambda, Lu, tol, maxiter, F_tol);
    
        F_amanpg = min(obj_arr_amanpg);
    
        %% ALMTR
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
    
        maxit_att_ALMTR = tr_solver.iter_num;
        time_arr_ALMTR = tr_solver.record.time;
        obj_arr_ALMTR = tr_solver.record.loss;
        gx_arr_ALMTR = [gx_arr_amanpg(2) tr_solver.record.gX+tr_solver.record.gU];
        fv0 = min(obj_arr_ALMTR);
    
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
    
        maxit_att_ALMSSN = almssn_solver.iter_num;
        time_arr_ALMSSN = [0 almssn_solver.record.time];
        obj_arr_ALMSSN = [obj_arr_amanpg(1) almssn_solver.record.loss]; % 其实没用那个初始值
        gx_arr_ALMSSN = [gx_arr_amanpg(2) almssn_solver.record.gX+almssn_solver.record.gU];
        fv1 = min(obj_arr_ALMSSN);
    
        % ================= LSq-II ==================
        option_almssn.LS = 2;
        option_almssn.x_init = Xinitial.main;
        option_almssn.algname = 'LSq-II';
        almssn_solver = SPCANewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);
    
        tic;
        almssn_solver = almssn_solver.run(tol);
        time2 = toc;
    
        maxit_att_ALMSSN2 = almssn_solver.iter_num;
        time_arr_ALMSSN2 = [0 almssn_solver.record.time];
        obj_arr_ALMSSN2 = [obj_arr_amanpg(1) almssn_solver.record.loss]; % 其实没用那个初始值
        gx_arr_ALMSSN2 = [gx_arr_amanpg(2) almssn_solver.record.gX+almssn_solver.record.gU];
        fv2 = min(obj_arr_ALMSSN2);
    
        type = 1;
    
        %% S0C
        option_soc.phi_init = phi_init; option_soc.maxiter = 30000;  option_soc.tol = 5e-8; %option_soc.dual_gap = 5e-7;
        option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
        option_soc.L= m;  option_soc.type = type;
        option_soc.F_mialm = -inf;
        option_soc.X_mialm = tr_solver.X;
        
        [X_Soc, F_SOC,sparsity_soc,time_soc,...
            error_XPQ, iter_soc, flag_succ,record_soc]= soc_spca(AtA,option_soc);
    
        time_arr_soc = [0 record_soc.time];
        obj_arr_soc=[obj_arr_amanpg(1) record_soc.loss];
        gx_arr_soc=[gx_arr_amanpg(2) record_soc.stops];
        F_soc = min(obj_arr_soc(10:iter_soc));
    
        %% ARPG
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
        
        [X_arpg, maxit_att_arpg, time_arpg, F_arpg,...
            nD5, sparsity5, avar5,time_arr_arpg,obj_arr_arpg,gx_arr_arpg] = ARPG_SPCA(Xinitial, A, lambda, Ll, Lu, tol, maxiter,  F_tol, RPMWHmaxiter, RPMWHtol);
    
    
        fmin = min([obj_arr_amanpg(maxit_att_amanpg),obj_arr_ALMTR(maxit_att_ALMTR),obj_arr_ALMSSN(maxit_att_ALMSSN),obj_arr_ALMSSN2(maxit_att_ALMSSN2),obj_arr_soc(iter_soc),obj_arr_arpg(maxit_att_arpg)]);
    
        figure(1)
        semilogy(time_arr_amanpg(1:10:maxit_att_amanpg), max(obj_arr_amanpg(1:10:maxit_att_amanpg) - fmin,eps),'color',colors(1,:), 'DisplayName', 'AManPG')
        hold on
        semilogy(time_arr_ALMTR(1:maxit_att_ALMTR), max(obj_arr_ALMTR(1:maxit_att_ALMTR) - fmin,eps),'color',colors(2,:), 'DisplayName', 'ALMSRTR')
        hold on
        semilogy(time_arr_ALMSSN(1:maxit_att_ALMSSN), max(obj_arr_ALMSSN(1:maxit_att_ALMSSN) - fmin,eps),'color',colors(3,:), 'DisplayName', 'ALMSSN\_LS1')
        hold on
        semilogy(time_arr_ALMSSN2(1:maxit_att_ALMSSN2), max(obj_arr_ALMSSN2(1:maxit_att_ALMSSN2) - fmin,eps),'color',colors(4,:), 'DisplayName', 'ALMSSN\_LS2')
        hold on
        semilogy(time_arr_soc(1,10:10:iter_soc), max(obj_arr_soc(1,10:10:iter_soc) - fmin,eps),'color',colors(5,:), 'DisplayName', 'SOC')
        hold on
        semilogy(time_arr_arpg(1:10:maxit_att_arpg), max(obj_arr_arpg(1:10:maxit_att_arpg) - fmin,eps),'color',colors(6,:), 'DisplayName', 'ARPG')
        hold on
        figure(2)
        semilogy(time_arr_amanpg(1:10:maxit_att_amanpg), gx_arr_amanpg(1:10:maxit_att_amanpg),'color',colors(1,:), 'DisplayName', 'AManPG')
        hold on
        semilogy(time_arr_ALMTR(1:maxit_att_ALMTR), gx_arr_ALMTR(1:maxit_att_ALMTR),'color',colors(2,:), 'DisplayName', 'ALMSRTR')
        hold on
        semilogy(time_arr_ALMSSN(1:maxit_att_ALMSSN), gx_arr_ALMSSN(1:maxit_att_ALMSSN),'color',colors(3,:), 'DisplayName', 'ALMSSN\_LS1')
        hold on
        semilogy(time_arr_ALMSSN2(1:maxit_att_ALMSSN2), gx_arr_ALMSSN2(1:maxit_att_ALMSSN2),'color',colors(4,:), 'DisplayName', 'ALMSSN\_LS2')
        hold on
        semilogy(time_arr_soc(1,10:10:iter_soc), gx_arr_soc(1,10:10:iter_soc) ,'color',colors(5,:), 'DisplayName', 'SOC')
        hold on
        semilogy(time_arr_arpg(1:10:maxit_att_arpg), gx_arr_arpg(1:10:maxit_att_arpg),'color',colors(6,:), 'DisplayName', 'ARPG')
        hold on
    
        figure(1)
    %     title_name = [num2str(m),'*', num2str(n),' ill-conditioned matrix'];
        xlabel('Time (secs.)'); ylabel('Shifted Loss')
    %     title(title_name,'Interpreter','none')
        set(gca,'XLim',[0 30],'YLim',[1e-8 1e3]);
        legend('location','southeast')
        hold off
        saveas(gcf,['figure\SPCA-' num2str(n) '-' num2str(num)],'fig')
    
        figure(2)
    %     title_name = [num2str(m),'*', num2str(n),' ill-conditioned matrix'];
        xlabel('Time (secs.)'); ylabel('Termination Condition') 
    %     title(title_name,'Interpreter','none')
        set(gca,'XLim',[0 30],'YLim',[1e-8 1e3]);
        legend('location','southeast')
        hold off
        saveas(gcf,['figure\SPCA-' num2str(n) '-' num2str(num) 'tc'],'fig')
        close all
        catch
            continue
        end
        
    end

end