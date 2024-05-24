clear;
close all;
addpath(genpath(pwd))
addpath ./manopt;

set(groot,'defaultLineLineWidth',2)

n = 500;
m = 50;
for r = [20,30,40]
    lambda = 0.05;
    start = 5;
    reps = 5;
    
    for num = start:reps
    
        seed = 1*num;
        rng(seed);
        colors = linspecer(8);
    
        dx = m/n;  V = 0;
        Lu = 8/dx^2.*(sin(pi/4))^2;
        Ll = Lu*0.8;
        A = -Sch_matrix(0,m,n); %  schrodinger operator
    
        [phi_init,~] = svd(randn(n,r),0);  % random intialization
    
        Xinitial.main = phi_init;
    
        dx = m/n;  V = 0;
    
    
        %% AMANPG
        Xinitial.main = phi_init;
    
        tol = 5e-5;
        maxiter = 30000;
        F_tol = -inf;%fv0;%out_mialm.obj;%-Inf;
    
        [X_amanpg, maxit_att_amanpg, time_amanpg, F_amanpg, ~, sparsity4, avar4, time_arr_amanpg,obj_arr_amanpg,gx_arr_amanpg] = AManPG_CM(Xinitial, A, lambda, Lu, tol, maxiter, F_tol);
    
        F_amanpg = min(obj_arr_amanpg);
    
        %% ALMTR
        option_tr.mu = lambda;
        option_tr.n = n; option_tr.r = r;
        option_tr.tau = 0.99;
        option_tr.sigma_factor = 1.25;
        if n>500
            option_tr.maxtr_iter = 40;
        else
            option_tr.maxtr_iter = 60;
        end
        option_tr.retraction = 'retr';
        option_tr.verbose = 0;
        option_tr.x_init = Xinitial.main;
        tr_solver = CMRTR();
        tr_solver = tr_solver.init(A, option_tr);
    
        tic;
        tr_solver = tr_solver.run(5.0e-7, tol);
        time0 = toc;
    
        maxit_att_ALMTR = tr_solver.iter_num;
        time_arr_ALMTR = [0 tr_solver.record.time];
        obj_arr_ALMTR = [obj_arr_amanpg(1) tr_solver.record.loss];
        gx_arr_ALMTR = [gx_arr_amanpg(2) tr_solver.record.gX+tr_solver.record.gU];
        fv0 = min(obj_arr_ALMTR);
        X_ALMTR = tr_solver.X;
    
    
    
        % ================= LSq-I ==================
        option_almssn.mu = lambda;
        option_almssn.n = n; option_almssn.r = r;
        option_almssn.tau = 0.97;
        option_almssn.retraction = 'retr';
        option_almssn.adap_maxiter = 0;
        option_almssn.sigma_factor = 1.25;
        option_almssn.maxcg_iter = 1000;
        %option_almssn.maxinner_iter = 300;
        option_almssn.maxinner_iter = 1000;
        option_almssn.maxnewton_iter = 40;
        %option_almssn.maxnewton_iter = 10;
        option_almssn.gradnorm_decay = 0.95;
        option_almssn.gradnorm_min = 1.0e-13;
        option_almssn.verbose = 0;
        option_almssn.LS = 1;
        option_almssn.x_init = Xinitial.main;
        option_almssn.algname = 'LSq-I';
        almssn_solver = CMNewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);
    
        tic;
        almssn_solver = almssn_solver.run(5.0e-7, tol);
        time0 = toc;
    
        maxit_att_ALMSSN = almssn_solver.iter_num;
        time_arr_ALMSSN = [0 almssn_solver.record.time];
        obj_arr_ALMSSN = [obj_arr_amanpg(1) almssn_solver.record.loss]; % 其实没用那个初始值
        gx_arr_ALMSSN = [gx_arr_amanpg(2) almssn_solver.record.gX+almssn_solver.record.gU];
        fv1 = min(obj_arr_ALMSSN);
        X_ALMSSN1 = almssn_solver.X;
    
        
    % 
    %     % ================= LSq-II ==================
        option_almssn.LS = 2;
        option_almssn.x_init = Xinitial.main;
        option_almssn.algname = 'LSq-II';
        almssn_solver = CMNewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);
    
        tic;
        almssn_solver = almssn_solver.run(5.0e-7, tol);
        time1 = toc;
    
        maxit_att_ALMSSN2 = almssn_solver.iter_num;
        time_arr_ALMSSN2 = [0 almssn_solver.record.time];
        obj_arr_ALMSSN2 = [obj_arr_amanpg(1) almssn_solver.record.loss]; % 其实没用那个初始值
        gx_arr_ALMSSN2 = [gx_arr_amanpg(2) almssn_solver.record.gX+almssn_solver.record.gU];
        fv2 = min(obj_arr_ALMSSN2);
        X_ALMSSN2 = almssn_solver.X;
    
    
        %% mialm
        type = 1;
    
        H = A;
    
    
        Op = struct();
        Op.applyA = @(X) X;
        Op.applyAT = @(y) y;
    
        f = struct();
        f.cost_grad = @cms_cost_grad;
        f.data = {H};
    
        h = struct();
        h.cost = @(X,lambda) lambda*sum(sum(abs(X)));
        h.prox = @(X,nu,lambda) max(abs(X) - nu*lambda,0).* sign(X);
        h.data = {lambda};
    
    
        manifold = stiefelfactory(n,r);
    
        options_mialm.max_iter = 30000;     options_mialm.maxitersub = 10;
        options_mialm.tau = 0.8;          options_mialm.rho = 1.05;
        options_mialm.nu0 = svds(H,1,'largest','MaxIterations',10000)^1*2 ; options_mialm.tol = 5e-5;
        options_mialm.gtol0 = 1;          options_mialm.gtol_decrease = 0.8;
        options_mialm.X0 = phi_init;      options_mialm.verbosity = 0;
        [X_mialm,Z_mialm,out_mialm] = mialm_CM(Op, manifold, f, h, options_mialm);
    
        gx_arr_mialm = out_mialm.gx_arr;
        time_arr_mialm = out_mialm.time_arr;
        obj_arr_mialm = out_mialm.obj_arr;
        iter_mialm = out_mialm.iter;
        F_mialm = min(out_mialm.obj_arr(1:iter_mialm+1));
    
    
    
        %% manpg_adap
        option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
        option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
        option_manpg.L = m; option_manpg.F_mialm = -inf;
        option_manpg.inner_iter = 100;
        option_manpg.global_tol = 5e-5;
    
        [X_manpg_BB, F_manpg_BB,sparsity_manpg_BB,time_manpg_BB,...
            maxit_att_manpg_BB,succ_flag_manpg_BB,lins_adap_manpg,in_av_adap_manpg,record_manpg_adap]= manpg_CMS_adap(H,option_manpg,dx,V);
    
        time_arr_manpg_BB = record_manpg_adap.time;
        obj_arr_manpg_BB = record_manpg_adap.loss;
        gx_arr_manpg_BB = record_manpg_adap.gX;
        F_manpg_BB = min(obj_arr_manpg_BB);
    
    
            
            
        %% S0C
        option_soc.phi_init = phi_init; option_soc.maxiter = 30000;  option_soc.tol = 5e-5; option_soc.dual_gap = 5e-7;
        option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
        option_soc.L= m;  option_soc.type = type;
        option_soc.F_mialm = -inf;
        option_soc.X_mialm = tr_solver.X;
    
        [X_Soc, F_soc,sparsity_soc,time_soc,...
            soc_error_XPQ,iter_soc, flag_succ,record_soc]= soc_CM(H,option_soc);
    
        time_arr_soc = [0 record_soc.time];
        obj_arr_soc=[obj_arr_amanpg(1) record_soc.loss];
        gx_arr_soc=[gx_arr_amanpg(2) record_soc.gP+record_soc.gQ+record_soc.gX];
        F_soc = min(obj_arr_soc(10:iter_soc-1));
    
    
    
        %% ARPG
        Xinitial.main = phi_init;
    
        tol = 5e-5;
        maxiter = 30000;
        %         RPMWHmaxiter = 20;
        %         RPMWHtol = 1e-3;
        RPMWHmaxiter = 50;
        RPMWHtol = 3e-3;
        F_tol = -inf;%fv0;%out_mialm.obj;%-inf;
    
        [X_arpg, maxit_att_arpg, time_arpg, F_arpg, nD5, sparsity5, avar5,time_arr_arpg,obj_arr_arpg,gx_arr_arpg] = ARPG_CM(Xinitial, A, lambda, Ll, Lu, tol, maxiter,  F_tol, RPMWHmaxiter, RPMWHtol);
    
        F_arpg = min(obj_arr_arpg);
        
        fmin = min([obj_arr_amanpg(maxit_att_amanpg),obj_arr_ALMTR(maxit_att_ALMTR),obj_arr_ALMSSN(maxit_att_ALMSSN),obj_arr_ALMSSN2(maxit_att_ALMSSN2),obj_arr_soc(iter_soc-1),obj_arr_arpg(maxit_att_arpg),obj_arr_mialm(iter_mialm+1),obj_arr_manpg_BB(maxit_att_manpg_BB-1)]);
    
        figure(1)
            semilogy(time_arr_amanpg([1:10:maxit_att_amanpg,maxit_att_amanpg]), max(obj_arr_amanpg([1:10:maxit_att_amanpg,maxit_att_amanpg]) - fmin,eps),'color',colors(1,:), 'DisplayName', 'AManPG')
            hold on
        figure(2)
            semilogy(time_arr_amanpg([1:10:maxit_att_amanpg,maxit_att_amanpg]), gx_arr_amanpg([1:10:maxit_att_amanpg,maxit_att_amanpg]),'color',colors(1,:), 'DisplayName', 'AManPG')
            hold on
        figure(1)
            semilogy(time_arr_ALMTR(1:maxit_att_ALMTR), max(obj_arr_ALMTR(1:maxit_att_ALMTR) - fmin,eps),'color',colors(2,:), 'DisplayName', 'ALMSRTR')
            hold on
        figure(2)
            semilogy(time_arr_ALMTR(1:maxit_att_ALMTR), gx_arr_ALMTR(1:maxit_att_ALMTR),'color',colors(2,:), 'DisplayName', 'ALMSRTR')
            hold on
        figure(1)
            semilogy(time_arr_ALMSSN(1:maxit_att_ALMSSN), max(obj_arr_ALMSSN(1:maxit_att_ALMSSN) - fmin,eps),'color',colors(3,:), 'DisplayName', 'ALMSSN\_LS1')
            hold on
        figure(2)
            semilogy(time_arr_ALMSSN(1:maxit_att_ALMSSN), gx_arr_ALMSSN(1:maxit_att_ALMSSN),'color',colors(3,:), 'DisplayName', 'ALMSSN\_LS1')
            hold on
        figure(1)
            semilogy(time_arr_ALMSSN2(1:maxit_att_ALMSSN2), max(obj_arr_ALMSSN2(1:maxit_att_ALMSSN2) - fmin,eps),'color',colors(4,:), 'DisplayName', 'ALMSSN\_LS2')
            hold on
        figure(2)
            semilogy(time_arr_ALMSSN2(1:maxit_att_ALMSSN2), gx_arr_ALMSSN2(1:maxit_att_ALMSSN2),'color',colors(4,:), 'DisplayName', 'ALMSSN\_LS2')
            hold on
        figure(1)
            semilogy(time_arr_mialm([1:10:iter_mialm+1,iter_mialm+1]), max(obj_arr_mialm([1:10:iter_mialm+1,iter_mialm+1]) - fmin,eps),'color',colors(5,:), 'DisplayName', 'MIALM')
            hold on
        figure(2)
            semilogy(time_arr_mialm([1:10:iter_mialm+1,iter_mialm+1]), gx_arr_mialm([1:10:iter_mialm+1,iter_mialm+1]) ,'color',colors(5,:), 'DisplayName', 'MIALM')
            hold on
        figure(1)
            semilogy(time_arr_manpg_BB([1:10:maxit_att_manpg_BB-1,maxit_att_manpg_BB-1]), max(obj_arr_manpg_BB([1:10:maxit_att_manpg_BB-1,maxit_att_manpg_BB-1]) - fmin,eps),'color',colors(6,:), 'DisplayName', 'ManPG\_Ada')
            hold on
        figure(2)
            semilogy(time_arr_manpg_BB([1:10:maxit_att_manpg_BB-1,maxit_att_manpg_BB-1]), gx_arr_manpg_BB([1:10:maxit_att_manpg_BB-1,maxit_att_manpg_BB-1]),'color',colors(6,:), 'DisplayName', 'ManPG\_Ada')
            hold on
        figure(1)
            semilogy(time_arr_soc([1,10:10:iter_soc-1,iter_soc-1]), max(obj_arr_soc([1,10:10:iter_soc-1,iter_soc-1]) - fmin,eps),'color',colors(7,:), 'DisplayName', 'SOC')
            hold on
        figure(2)
            semilogy(time_arr_soc([1,10:10:iter_soc-1,iter_soc-1]), gx_arr_soc([1,10:10:iter_soc-1,iter_soc-1]) ,'color',colors(7,:), 'DisplayName', 'SOC')
            hold on
        figure(1)
            semilogy(time_arr_arpg([1:10:maxit_att_arpg,maxit_att_arpg]), max(obj_arr_arpg([1:10:maxit_att_arpg,maxit_att_arpg]) - fmin,eps),'color',colors(8,:), 'DisplayName', 'ARPG')
            hold on
        figure(2)
            semilogy(time_arr_arpg([1:10:maxit_att_arpg,maxit_att_arpg]), gx_arr_arpg([1:10:maxit_att_arpg,maxit_att_arpg]),'color',colors(8,:), 'DisplayName', 'ARPG')
            hold on
    
    
        
        figure(1) 
    %         title_name = [num2str(m),'*', num2str(n),' ill-conditioned matrix'];
            xlabel('Time (secs.)'); ylabel('Shifted Loss')
    %         title(title_name,'Interpreter','none')
            set(gca,'XLim',[0 30],'YLim',[1e-8 1e3]);
            legend('location','northeast')
            hold off
            saveas(gcf,['figure\CM' num2str(r) '-' num2str(num)],'fig')
        figure(2)
    %         title_name = [num2str(m),'*', num2str(n),' ill-conditioned matrix'];
            xlabel('Time (secs.)'); ylabel('Termination Condition') 
    %         title(title_name,'Interpreter','none')
            set(gca,'XLim',[0 30],'YLim',[1e-6 1e3]);
            legend('location','northeast')
            hold off
            saveas(gcf,['figure\CM' num2str(r) '-' num2str(num) 'tc'],'fig')
    
        
        fid = 1;
        fprintf(fid,' ---------------------------------------------------------\n');
        H =  - H;
        [VV,D] = eig(H+0);
        D = diag(D);
        [D I]= sort(D,'ascend');
    
        %flip the negative columns of CMs to be positive
        X_amanpg = positive_CMs(X_amanpg,r);
        X_ALMTR = positive_CMs(X_ALMTR,r);
        X_ALMSSN1 = positive_CMs(X_ALMSSN1,r);
        X_ALMSSN2 = positive_CMs(X_ALMSSN2,r);
        X_mialm = positive_CMs(X_mialm,r);
        X_manpg_BB = positive_CMs(X_manpg_BB,r);
        X_Soc = positive_CMs(X_Soc,r);
        X_arpg = positive_CMs(X_arpg,r);
    
        % eigens of phi'*H*phi- manpg-adap
        
        [U1 D1] = eig(X_amanpg'*H*X_amanpg);
        D1 = diag(D1);
        [D1 ind1] = sort(D1,'ascend');
        U1 = U1(:,ind1);
        [U2 D2] = eig(X_ALMTR'*H*X_ALMTR);
        D2 = diag(D2);
        [D2 ind2] = sort(D2,'ascend');
        [U3 D3] = eig(X_ALMSSN1'*H*X_ALMSSN1);
        D3 = diag(D3);
        [D3 ind3] = sort(D3,'ascend');
        [U4 D4] = eig(X_ALMSSN2'*H*X_ALMSSN2);
        D4 = diag(D4);
        [D4 ind4] = sort(D4,'ascend');
        U4 = U4(:,ind4);
        [U5 D5] = eig(X_mialm'*H*X_mialm);
        D5 = diag(D5);
        [D5 ind5] = sort(D5,'ascend');
        U5 = U5(:,ind5);
        [U6 D6] = eig(X_manpg_BB'*H*X_manpg_BB);
        D6 = diag(D6);
        [D6 ind6] = sort(D6,'ascend');
        U6 = U6(:,ind6);
        [U7 D7] = eig(X_Soc'*H*X_Soc);
        D7 = diag(D7);
        [D7 ind7] = sort(D7,'ascend');
        U7 = U7(:,ind7);
        [U8 D8] = eig(X_arpg'*H*X_arpg);
        D8 = diag(D8);
        [D8 ind8] = sort(D8,'ascend');
        U8 = U8(:,ind8);
        figure(3);
       % set(gcf,'Position',[324   455   905   322]);
        plot(D(1:r),'g*','LineWidth',1); hold on;
        plot(D1(1:r),'ko','LineWidth',1,'MarkerSize',10);
        hold on;
        plot(D2(1:r),'bd','LineWidth',1)
        hold on;
        plot(D3(1:r),'c+','LineWidth',1)
        hold on;
        plot(D4(1:r),'rx','LineWidth',1)
        hold on;
        plot(D5(1:r),'m^','LineWidth',1)
        hold on;
        plot(D6(1:r),'Color','#EDB120','Marker','v','LineWidth',1)
        hold on;
        plot(D7(1:r),'Color','#4DBEEE','Marker','hexagram','LineWidth',1)
        hold on;
        plot(D8(1:r),'Color','#A2142F','Marker','pentagram','LineWidth',1)
        
        set(gcf, 'PaperPositionMode', 'auto');
        hlen = legend('Groundtruth','AmanPG','ALMSRTR','ALMSSN\_LS1','ALMSSN\_LS2','MIALM','ManPG','SOC','ARPG','Location','northwest');
        set(hlen,'FontSize',10,'Interpreter','latex');
        set(gca,'FontSize',10);
        set(gca,'XLim',[0 r],'YLim',[0 1]);
        xlabel('k'); ylabel('k-th Eigenvalue')
        saveas(gcf,['figure\CM' num2str(r) '-' num2str(num) 'eig'],'fig')
        close all
    
    end
end

function [f,g] = cms_cost_grad(X,AtA)
        BX = AtA*X;
        f = -sum(sum(BX.*X));
        g = -2*BX;
end