%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Developed by Michael Santos Duarte
%PPGETI
%Out 12, 2018
%
%Description:
% Comparison of the results of the OS-CKL algorithm with the novelty criterion and
% pruning
%OS-CKL (Online Sparse Correntropy Kernel Learning)
%NCL1 (Novelty Criterion with [or adaptive] L1-norm regularization function)
%Algorithm type: Recursive
%Dataset: Wiener-Hammerstein
%One step ahead
%Multi step ahead
%Nonlinear Auoregressive Ex (NARX)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all, clear all
clc

%%


%======data formatting===========
% data settings
Nt  = 168000; % number of samples 188,000

load('WienerHammerBenchMark.mat')
 
uu=uBenchMark(10000:188000-10000)';  % select the section of the experiment
yy=yBenchMark(10000:188000-10000)';

%======data embedding=======
na = 10; % Order output
nb = 10; % Order input
d = 0; % Delay
nu = na+nb+1; % Parameters
x(:,1) = zeros(nu,1);
%data size
N_tr = 70000;
N_val = 48000;
N_te = 50000;%
%%======end of data=======

for i = 2:Nt
      
    for j=1:na
        if i-j <= 0
            x(j,i) = 0;
        else
            x(j,i) = -yy(i-j);
        end
    end
    for j=0:nb
        if i-j-d <= 0
            x(j+1+na,i) = 0;
        else
            x(j+1+na,i) = uu(i-j-d);
        end
    end
end

%======noise================
% pdf do ruído (Mixtura de ruídos impulsivos)
eps = [0 0.05];
t2 = N_tr; % Quantidade de amostras
m1 = -.1;
m2 = .2;
s1 = .06;
s2 = .06;
%s2 = 3*var(yy); %Desvio da gaussiana 2
Z = zeros(t2,length(eps)); %Inicializa variavel do ruído
nOutlier = length(eps);
% for no = 1:nOutlier
%     alpha = 1-eps(no); %Shape da mixtura de gaussiana
%     U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
%     I1 = (U < alpha); %Define o shape do ruido
%     Z(:,no) = I1.*(randn(t2,1)*s1 + m1) + (1-I1).*(randn(t2,1)*s2 + m2); %Gera o ruido da mistura de gaussianas
% end

% % Define the distribution parameters (means and covariances) of a two-component bivariate Gaussian mixture distribution. 
% mu = [1 2;-3 -5];
% sigma = [1 1]; % shared diagonal covariance matrix
% 
% % Create a gmdistribution object by using the gmdistribution function. By default, the function creates an equal proportion mixture.
% gm = gmdistribution(mu,sigma);

%%======end noise=======

% training data
trainInput = x(:,50001:120000);
trainTarget = yy(50001:120000)';

% validate data
valInput = x(:,120001:Nt);
valTarget = yy(120001:Nt)';

% testing data
testInput = x(:,1:N_te);
testTarget = yy(1:N_te)';
testTargetdefault = yy(1:N_te)';

% Data size for training and testing
trainSize = length(trainTarget);
valSize = length(valTarget);
testSize = length(testTarget);

%======end of data formatting===========

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             SIMULATIONS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .01;
nTrial = 1; % Number of simulations
trainTarget0 = trainTarget;

for no = 1:nOutlier
    
    for nt = 1:nTrial
        
        alpha = 1-eps(no); %Shape da mixtura de gaussiana
        U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
        I1 = (U < alpha); %Define o shape do ruido
        
        %z = random(gm,t2);
        %zz = [z(:,1); z(:,2)];
        
%         m1 = 0;
%         m2 = .2;
%         s1 = .06;
%         s2 = .4;  
        %Z(:,no) = (1-I1).*zz; %Gera o ruido da mistura de gaussianas
%         Z(:,no) = I1.*0 + (1-I1).*((randn(t2,1)*s1 + m1) + (randn(t2,1)*s2 + m2)); %Gera o ruido da mistura de gaussianas
        %Z(:,no) = I1.*0 + (1-I1).*(var(yy)*trnd(2)); %Gera o ruido da mistura de t-student
        Z(:,no) = I1.*0 + (1-I1).*random('poiss', 0.05, t2, 1);
        
%         alpha = 1-eps(no); %Shape da mixtura de gaussiana
%         U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
%         I1 = (U < alpha); %Define o shape do ruido
%         
%         Z(:,no) = Z(:,no) + I1.*0 + -(1-I1).*random('poiss', 0.05, t2, 1);

        trainTarget = trainTarget0+Z(:,no);
        
          figure, plot(trainTarget), hold on, plot(trainTarget0)
        
        disp(['Outlier # ',num2str(no.'),'/',num2str(nOutlier), '|| Trial # ',num2str(nt.'),'/',num2str(nTrial)]);
              
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-NCL1
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        length_nc = 1;
        
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;

        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_distance_nc_vector = .6;%linspace(1.6,1.6,length_nc); %1.6%.6%.8
            th_error_nc_vector  = 0.0196;%linspace(0.0196,0.0196,length_nc);
            %th_distance_nc_vector = .006;%.006; Poly = 3; 0.005; ; Gauss=2;
            %th_error_nc_vector = .001;%.001; Poly = 3; 0.01; Gauss=2;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 11; % Regularization parameter correntropy
        else
            th_distance_nc_vector = 1.2;%linspace(1.8,1.8,length_nc);%1.8%1.2%.03095
            th_error_nc_vector  = 0.0206;%linspace(0.0206,0.0206,length_nc);
%             th_distance_nc_vector = .008;%.009; Poly = 3
%             th_error_nc_vector = .003;%.004; Poly = 3
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 20; % Regularization parameter correntropy
        end
        %th_ald_vector = .1;
        mse_cskl_nc = zeros(length_nc, 1);
        distsize_cskl_nc = zeros(length_nc, 1);

        fprintf('Learning OS-CKL-NCL1...\n');

        for ii = 1:length_nc 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_nc)]);

            th_distance_nc = th_distance_nc_vector(ii);
                th_error_nc = th_error_nc_vector(ii);
                
            [expansionCoefficient2,biasCoefficient2,dictionaryIndex2,learningCurve2,identOutlier2] = ...
                OSCKL_NCL1(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th_distance_nc,th_error_nc_vector,.005,flagLearningCurve,ks,epson_updt);

            % ONE STEP AHEAD
            y_teCSKLnc = zeros(testSize,1);
            for jj = 1:testSize
                y_teCSKLnc(jj) = expansionCoefficient2'*...
                    ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel) + biasCoefficient2(1);
            end
            outlier(no).trial(nt).identOutliernc(:,ii)  = identOutlier2;
            outlier(no).trial(nt).distsize_cskl_nc(ii) = length(dictionaryIndex2);
            outlier(no).trial(nt).mse_cskl_nc(ii) = mean((testTarget - y_teCSKLnc).^2);

            % MULT STEP AHEAD                
            y_kte_csklnc = zeros(testSize,1);
            
            xkt(:,1) = zeros(nu,1);

            for jj = 1:testSize
                y_kte_csklnc(jj) = expansionCoefficient2'*ker_eval(xkt(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel) + biasCoefficient2(1);

                yp(jj) = y_kte_csklnc(jj);
                
                for j=1:na
                    if jj-j <= 0
                        xkt(j,jj+1) = 0;
                    else
                        xkt(j,jj+1) = -yp(jj+1-j);
                    end
                end
                for j=0:nb
                    if jj-j-d <= 0
                        xkt(j+1+na,jj+1) = 0;
                    else
                        xkt(j+1+na,jj+1) = uu(jj+1-j-d);
                    end
                end
            end

            outlier(no).trial(nt).rmse_csklnc_msa(ii) = sqrt(mean((testTarget - y_kte_csklnc).^2)); % RMSE

            error_csklnc = testTarget - y_kte_csklnc;
            outlier(no).trial(nt).error.msa.csklnc(:,ii) = error_csklnc;
            sig = 1;
            outlier(no).trial(nt).c_csklnc(ii) = mean(exp(-((error_csklnc).^2)./(2*sig^2))); % Correntropy

            outlier(no).trial(nt).output.osa.csklnc(:,ii) = y_teCSKLnc;
            outlier(no).trial(nt).output.msa.csklnc(:,ii) = y_kte_csklnc;

        end

        % =========end of OS-CKL-NCL1================
        
    outlier(no).trial(nt).norma = norm(expansionCoefficient2);

        
    end % end nTrial
        
end % end nOutlier

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              RESULTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p_outlier = [0 5];

for no = 1:nOutlier
    
    for nt = 1:nTrial        
        best.csklnc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklnc)==outlier(no).trial(nt).c_csklnc,1);
        best.csklnc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_nc(best.csklnc.index.outlier(no).trial(nt)));
        best.csklnc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklnc_msa(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklnc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_nc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliernc(:,best.csklnc.index.outlier(no).trial(nt));    
        best.csklnc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt)); 
    end
    
    best.csklnc.osa.rmse.mean.outlier(no) = mean(best.csklnc.osa.rmse.outlier(no).trial);
    best.csklnc.msa.rmse.mean.outlier(no) = mean(best.csklnc.msa.rmse.outlier(no).trial);
    best.csklnc.msa.ctp.mean.outlier(no) = mean(best.csklnc.msa.ctp.outlier(no).trial);
    best.csklnc.osa.rmse.std.outlier(no) = std(best.csklnc.osa.rmse.outlier(no).trial);
    best.csklnc.msa.rmse.std.outlier(no) = std(best.csklnc.msa.rmse.outlier(no).trial);
    best.csklnc.msa.ctp.std.outlier(no) = std(best.csklnc.msa.ctp.outlier(no).trial);
    best.csklnc.index.best.outlier(no) = find(max(best.csklnc.msa.ctp.outlier(no).trial)==best.csklnc.msa.ctp.outlier(no).trial,1);
    best.csklnc.osa.output.data.outlier(:,no) = best.csklnc.osa.output.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.msa.output.data.outlier(:,no) = best.csklnc.msa.output.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.msa.output.data.identoutlier(:,no) = best.csklnc.msa.output.identoutlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.sizeDic.min.outlier(no) = best.csklnc.sizeDic.outlier(no).trial(best.csklnc.index.best.outlier(no));
    best.csklnc.msa.rmse.best.outlier(no) = best.csklnc.msa.rmse.outlier(no).trial(best.csklnc.index.best.outlier(no));
    best.csklnc.sizeDic.mean.outlier(no) = mean(best.csklnc.sizeDic.outlier(no).trial);
    best.csklnc.sizeDic.std.outlier(no) = var(best.csklnc.sizeDic.outlier(no).trial);   
    best.csklnc.msa.error.data.outlier(:,no) = best.csklnc.msa.error.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.norma.outlier(no) =  outlier(no).trial(best.csklnc.index.best.outlier(no)).norma;  

end

% Norma alfa
best.csklnc.norma.outlier

% RMSE - ONE STEP AHEAD
figure
hold on
errorbar(p_outlier,best.csklnc.osa.rmse.mean.outlier,best.csklnc.osa.rmse.std.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - One step ahead');

% RMSE - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.csklnc.msa.rmse.mean.outlier,best.csklnc.msa.rmse.std.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
box on
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.csklnc.msa.rmse.mean.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Plot');

figure
hold on
plot(p_outlier,best.csklnc.msa.rmse.best.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Best');

% CORRENTROPY - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100,best.csklnc.msa.ctp.std.outlier.*100, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Plot');

% DICTIONARY
figure
hold on
dic = [best.csklnc.sizeDic.min.outlier];
bar(p_outlier,dic')
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
title('Min Dictionary of support vectors');

figure
hold on
dic2 = [best.csklnc.sizeDic.mean.outlier];
bar(p_outlier,dic2')
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
box on
title('Mean Dictionary of support vectors');


% Testing MSA
figure;
subplot(2,1,1)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1]);
xlabel('time (steps)')
ylabel('y')
% title('Predict output without outliers - OS-CKL-NCL1');
hold off;

% Testing MSA
subplot(2,1,2)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-NCL1');
hold off;

figure;
subplot(2,1,1)
range = 32250:32500;
plot(range,testTarget(32250:32500),'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(range,testTargetdefault(32250:32500),'m--','LineWidth',1);
plot(range, best.csklnc.msa.output.data.outlier(32250:32500,1),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
axis([32250 32500 -1 .5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers');
hold off;

subplot(2,1,2)
range = 32250:32500;
plot(range,testTarget(32250:32500),'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(range,testTargetdefault(32250:32500),'m--','LineWidth',1);
plot(range, best.csklnc.msa.output.data.outlier(32250:32500,2),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
axis([32250 32500 -1 .5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers');
hold off;

% Error testing MSA
figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,1) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
title('Error predict output without outliers - OS-CKL-NCL1');
hold off;

figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,nOutlier) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
title('Error predict output with 5% outliers - OS-CKL-NCL1');
hold off;

y = zeros(testSize,1);
for jj = 1:trainSize
    y(jj) = expansionCoefficient2'*...
        ker_eval(trainInput(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel) + biasCoefficient2(1);
end


ir = trainSize-2000;
er = trainSize+2000;
range = ir:er;
l = linspace(-1,1.3,length(range));
vl = ones(length(range),1).*trainSize;

figure,
plot(range,[trainTarget(end-2000:end); testTarget(1:2000)],'b','MarkerSize',15,'LineWidth',1);
hold on;
% plot(range,[trainTarget0(end-2000:end); testTargetdefault(1:1000)],'m--','LineWidth',1);
plot(range,[y(end-2000:end); best.csklnc.msa.output.data.outlier(1:2000,2)],'r','LineWidth',1);
% plot(vl,l,'--k')
legend('Desired','Simulated');
axis([ir er -1.5 1.5]);
xlabel('time (steps)')
ylabel('y (output)')
title('Training Simulation');
hold off;

% %%
% 
% samples = [trainTarget(end-3000:end); testTarget(1:2000)];
% desired = [trainTarget0(end-3000:end); testTargetdefault(1:2000)];
% simulated = [y(end-3000:end); best.csklnc.msa.output.data.outlier(1:2000,2)];
% 
% h = figure
% 
% ir = trainSize-3000;
% er = trainSize+2000;
% 
% grid on;
% range = ir:er;
% markers = animatedline('Marker','.','MaximumNumPoints', length(range),'Color','g');
% p = animatedline('Marker','.','MaximumNumPoints', length(range),'Color','b');
% p1 = animatedline('Marker','.','MaximumNumPoints', length(range),'Color','m');
% p2 = animatedline('Marker','.','MaximumNumPoints', length(range),'Color','r');
% axis([ir er -1 1.3])
% view(2);
% legend('End training','Samples','Desired','Simulated');
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
% l = linspace(-1,1.3,length(range));
% vl = ones(length(range),1).*trainSize;
% 
% addpoints(markers,vl,l);
% drawnow
% 
% for i=1:length(range)
% 
%    addpoints(p,range(i),samples(i));
%    addpoints(p1,range(i),desired(i));
%    addpoints(p2,range(i),simulated(i));
%    drawnow
% %    pause(.01)
%    
%       % Capture the plot as an image 
%       frame = getframe(h); 
%       im = frame2im(frame); 
%       [imind,cm] = rgb2ind(im,256); 
%       % Write to the GIF File 
%       if i == 1 
%           imwrite(imind,cm,'result.gif','gif', 'Loopcount',inf); 
%       else 
%           imwrite(imind,cm,'result.gif','gif','WriteMode','append'); 
%       end 
% end

for i = 1:nTrial
    sd1(i) = outlier(2).trial(i).distsize_cskl_nc(ii);
end
std(sd1)