%% AMATH 482 Homework 4
% Seong Hyun Han
% 3/06/20

%% Test 1 
clear all; close all; clc
% bg_name = band/genre name ('flume'), num_song = number of songs (30),
% ds = downsample number (4),
num_song = 30;
ds = 4;
[flume_data] = song_compiler('flume', num_song, ds);
[hall_data] = song_compiler('hall', num_song, ds);
[nat_data] = song_compiler('nat', num_song, ds);
[test1_data] = song_compiler('test1', num_song, ds);

%% Create Gabor transform spectrogram
music_time = 5; % 5 seconds for each audio clip
fs = 44100;
fs_d = 44100/ds;
L = music_time; 
tslide=0:0.1:L; % tau = 0.1 
n = fs_d * music_time; % Number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % Define the time domain discretization
t=t2(1:n); % Only consider the first n points for periodicity 
% Fourier components rescaled to have frequency in hertz 
k=(1/L)*[0:(n-1)/2 (-n+1)/2:-1]; % account for n being odd
ks=fftshift(k); % Fourier components with zero at the center 

%% Spectrogram data matrix for flume, hall, nat  
% a = 50, tau = 0.1, L = music duration, n = number of datapoints
% file_num = number of songs per data set, t = discrete time domain 
[flume_sg] = spectro(flume_data, 50, 0.1, L, n, num_song, t);
[hall_sg] = spectro(hall_data, 50, 0.1, L, n, num_song, t);
[nat_sg] = spectro(nat_data, 50, 0.1, L, n, num_song, t);
[test1_sg] = spectro(test1_data, 50, 0.1, L, n, num_song, t);

%% SVD + LDA
[U,S,V,w,threshold_1,threshold_2,sort_bg_1,sort_bg_2,sort_bg_3,~,~,~] = trainer(flume_sg,hall_sg,nat_sg,40);

%% Classify and Plots
% Plot singular values
figure(1)
subplot(1,3,1)
semilogy(diag(S),'ko','Linewidth',2)
hold on
diag_s = diag(S);
plot([0 length(diag(S))+1],[diag_s(40), diag_s(40)],'b','LineWidth',2)
set(gca,'Xlim',[0,length(diag(S))+1],'Fontsize',12)
title('Singular Values of the SVD of the Song Data','Fontsize',12)
xlabel('Singular Values','Fontsize',12)
ylabel('Non-normalized Energy','Fontsize',12)

% Plot song projections onto w
% figure(2)
% plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
% hold on
% plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
% plot(sort_bg_3,3.*ones(length(sort_bg_3)),'vg','Linewidth',2)
% ylim([0 4])
% title('Song Projections onto W','Fontsize',12)
% xlabel('Relative song location on W','Fontsize',12)

% Plot histogram of training data with thresholds
subplot(1,3,3)
histogram(sort_bg_1,length(sort_bg_1)) 
hold on 
histogram(sort_bg_2,length(sort_bg_2))
histogram(sort_bg_3,length(sort_bg_3)) 
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
title('Histogram of Training Data with Thresholds','Fontsize',12)
legend('Flume', 'Hall & Oats','Nat King Cole','Threshold 1','Threshold 2')
xlabel('Relative song location on W')
ylabel('Number of songs')
set(gca,'Fontsize',12)

% Classify 
TestNum = size(test1_sg,2); % 30
TestMat = U'*test1_sg;  % PCA projection
pval = w'*TestMat;  % LDA projection

% Plot test songs onto song data projections onto w
subplot(1,3,2)
plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
hold on
plot(pval(1:10),ones(length(pval(1:10))),'om','Linewidth',2)
plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
plot(pval(11:20),2.*ones(length(pval(11:20))),'dm','Linewidth',2)
plot(sort_bg_3, 3.*ones(length(sort_bg_2)),'vg','Linewidth',2)
plot(pval(21:30),3.*ones(length(pval(21:30))),'vm','Linewidth',2)
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
ylim([0 4])
title('Training and Test Song Projections onto W')
xlabel('Relative song location on W')
set(gca,'Fontsize',12)

% Find accuracy 
flume_test = pval(1:10);
hall_test = pval(11:20);
nat_test = pval(21:30);

% Accuracy of Nat King Cole
true_high = 0;
for i = 1:10
    if nat_test(i)>threshold_2
        true_high = true_high + 1;
    end
end
accuracy_nat = true_high/length(nat_test);

% Accuracy of Hall & Oats
true_mid = 0;
for i = 1:10
    if hall_test(i)>threshold_1 && hall_test(i)<threshold_2 
        true_mid = true_mid + 1;
    end
end
accuracy_hall = true_mid/length(hall_test);

% Accuracy of Flume
true_low = 0;
for i = 1:10
    if flume_test(i)<threshold_1
        true_low = true_low + 1;
    end
end
accuracy_flume = true_low/length(flume_test);

accuracy_total = (accuracy_nat + accuracy_hall + accuracy_flume)/3;
%% Test 2 
clear all; close all; clc 
num_song = 30;
ds = 4;
[bb_data] = song_compiler('bb', num_song, ds);
[bts_data] = song_compiler('bts', num_song, ds);
[gg_data] = song_compiler('gg', num_song, ds);
[test2_data] = song_compiler('test2', num_song, ds);

%% Create Gabor transform spectrogram
music_time = 5; % 5 seconds for each audio clip
fs = 44100;
fs_d = 44100/ds;
L = music_time; 
tslide=0:0.1:L; % tau = 0.1 
n = fs_d * music_time; % Number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % Define the time domain discretization
t=t2(1:n); % Only consider the first n points for periodicity 
% Fourier components rescaled to have frequency in hertz 
k=(1/L)*[0:(n-1)/2 (-n+1)/2:-1]; % account for n being odd
ks=fftshift(k); % Fourier components with zero at the center 

%% Spectrogram data matrix for flume, hall, nat  
% a = 50, tau = 0.1, L = music duration, n = number of datapoints
% file_num = number of songs per data set, t = discrete time domain 
[bb_sg] = spectro(bb_data, 50, 0.1, L, n, num_song, t);
[bts_sg] = spectro(bts_data, 50, 0.1, L, n, num_song, t);
[gg_sg] = spectro(gg_data, 50, 0.1, L, n, num_song, t);
[test2_sg] = spectro(test2_data, 50, 0.1, L, n, num_song, t);

%% SVD + LDA
[U,S,V,w,threshold_1,threshold_2,sort_bg_1,sort_bg_2,sort_bg_3,~,~,~] = trainer(bb_sg,bts_sg,gg_sg,40);

%% Classify and Plots
% Plot singular values
figure(1)
subplot(1,3,1)
semilogy(diag(S),'ko','Linewidth',2)
hold on
diag_s = diag(S);
plot([0 length(diag(S))+1],[diag_s(40), diag_s(40)],'b','LineWidth',2)
set(gca,'Xlim',[0,length(diag(S))+1],'Fontsize',12)
title('Singular Values of the SVD of the Song Data','Fontsize',12)
xlabel('Singular Values','Fontsize',12)
ylabel('Non-normalized Energy','Fontsize',12)

% Plot song projections onto w
% figure(2)
% plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
% hold on
% plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
% plot(sort_bg_3,3.*ones(length(sort_bg_3)),'vg','Linewidth',2)
% ylim([0 4])
% title('Song Projections onto W','Fontsize',12)
% xlabel('Relative song location on W','Fontsize',12)

% Plot histogram of training data with thresholds
subplot(1,3,3)
histogram(sort_bg_1,length(sort_bg_1)) 
hold on 
histogram(sort_bg_2,length(sort_bg_2))
histogram(sort_bg_3,length(sort_bg_3)) 
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
title('Histogram of Training Data with Thresholds','Fontsize',12)
legend('BIGBANG', 'BTS','Girls Generation','Threshold 1','Threshold 2')
xlabel('Relative song location on W')
ylabel('Number of songs')
set(gca,'Fontsize',12)

% Classify 
TestNum = size(test2_sg,2); % 30
TestMat = U'*test2_sg;  % PCA projection
pval = w'*TestMat;  % LDA projection

% Plot test songs onto song data projections onto w
subplot(1,3,2)
plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
hold on
plot(pval(1:10),ones(length(pval(1:10))),'om','Linewidth',2)
plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
plot(pval(11:20),2.*ones(length(pval(11:20))),'dm','Linewidth',2)
plot(sort_bg_3, 3.*ones(length(sort_bg_2)),'vg','Linewidth',2)
plot(pval(21:30),3.*ones(length(pval(21:30))),'vm','Linewidth',2)
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
ylim([0 4])
title('Training and Test Song Projections onto W')
xlabel('Relative song location on W')
set(gca,'Fontsize',12)

% Find accuracy 
bb_test = pval(1:10);
bts_test = pval(11:20);
gg_test = pval(21:30);

% Accuracy of BIGBANG
true_high = 0;
for i = 1:10
    if bb_test(i)>threshold_2
        true_high = true_high + 1;
    end
end
accuracy_bb = true_high/length(bb_test);

% Accuracy of Girls Generation
true_mid = 0;
for i = 1:10
    if gg_test(i)>threshold_1 && gg_test(i)<threshold_2 
        true_mid = true_mid + 1;
    end
end
accuracy_gg = true_mid/length(gg_test);

% Accuracy of BTS
true_low = 0;
for i = 1:10
    if bts_test(i)<threshold_1
        true_low = true_low + 1;
    end
end
accuracy_bts = true_low/length(bts_test);

accuracy_total = (accuracy_bb + accuracy_gg + accuracy_bts)/3;

%% Test 3
clear all; close all; clc 
num_song = 30;
ds = 4;
[classical_data] = song_compiler('classical', num_song, ds);
[punk_data] = song_compiler('punk', num_song, ds);
[rap_data] = song_compiler('rap', num_song, ds);
[test3_data] = song_compiler('test3', num_song, ds);

%% Create Gabor transform spectrogram
music_time = 5; % 5 seconds for each audio clip
fs = 44100;
fs_d = 44100/ds;
L = music_time; 
tslide=0:0.1:L; % tau = 0.1 
n = fs_d * music_time; % Number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % Define the time domain discretization
t=t2(1:n); % Only consider the first n points for periodicity 
% Fourier components rescaled to have frequency in hertz 
k=(1/L)*[0:(n-1)/2 (-n+1)/2:-1]; % account for n being odd
ks=fftshift(k); % Fourier components with zero at the center 

%% Spectrogram data matrix for flume, hall, nat  
% a = 50, tau = 0.1, L = music duration, n = number of datapoints
% file_num = number of songs per data set, t = discrete time domain 
[classical_sg] = spectro(classical_data, 50, 0.1, L, n, num_song, t);
[punk_sg] = spectro(punk_data, 50, 0.1, L, n, num_song, t);
[rap_sg] = spectro(rap_data, 50, 0.1, L, n, num_song, t);
[test3_sg] = spectro(test3_data, 50, 0.1, L, n, num_song, t);

%% SVD + LDA
[U,S,V,w,threshold_1,threshold_2,sort_bg_1,sort_bg_2,sort_bg_3,~,~,~] = trainer(punk_sg,classical_sg,rap_sg,40);

%% Plots

% Plot singular values
figure(1)
semilogy(diag(S),'ko','Linewidth',2)
set(gca,'Xlim',[0,length(diag(S))+1],'Fontsize',12)
title('Singular Values of the SVD of the Song Data')
xlabel('Singular Values')
ylabel('Non-normalized Energy')

% Plot song projections onto w
figure(2)
plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
hold on
plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
plot(sort_bg_3,3.*ones(length(sort_bg_3)),'*g','Linewidth',2)
ylim([0 4])
title('Song Projections onto W')
xlabel('Relative song location on W')

% Plot histogram of training data with thresholds
figure(3)
histogram(sort_bg_1,length(sort_bg_1)) 
hold on 
histogram(sort_bg_2,length(sort_bg_2))
histogram(sort_bg_3,length(sort_bg_3)) 
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
title('Classical, Punk, Rap')
set(gca,'Fontsize',12)
legend('Punk','Classical','Rap','Threshold 1','Threshold 2')
title('Histogram of Training Data with Thresholds')
ylabel('Number of songs')
xlabel('Relative song location on W')

%% Classify and Plots
% Plot singular values
figure(1)
subplot(1,3,1)
semilogy(diag(S),'ko','Linewidth',2)
hold on
diag_s = diag(S);
plot([0 length(diag(S))+1],[diag_s(40), diag_s(40)],'b','LineWidth',2)
set(gca,'Xlim',[0,length(diag(S))+1],'Fontsize',12)
title('Singular Values of the SVD of the Song Data','Fontsize',12)
xlabel('Singular Values','Fontsize',12)
ylabel('Non-normalized Energy','Fontsize',12)

% Plot song projections onto w
% figure(2)
% plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
% hold on
% plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
% plot(sort_bg_3,3.*ones(length(sort_bg_3)),'vg','Linewidth',2)
% ylim([0 4])
% title('Song Projections onto W','Fontsize',12)
% xlabel('Relative song location on W','Fontsize',12)

% Plot histogram of training data with thresholds
subplot(1,3,3)
histogram(sort_bg_1,length(sort_bg_1)) 
hold on 
histogram(sort_bg_2,length(sort_bg_2))
histogram(sort_bg_3,length(sort_bg_3)) 
plot([threshold_1 threshold_1],[0 6],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 6],'b','LineWidth',2)
title('Histogram of Training Data with Thresholds','Fontsize',12)
legend('Punk','Classical','Rap','Threshold 1','Threshold 2')
xlabel('Relative song location on W')
ylabel('Number of songs')
set(gca,'Fontsize',12)

% Classify 
TestNum = size(test3_sg,2); % 30
TestMat = U'*test3_sg;  % PCA projection
pval = w'*TestMat;  % LDA projection

% Plot test songs onto song data projections onto w
subplot(1,3,2)
plot(sort_bg_1,ones(length(sort_bg_1)),'ob','Linewidth',2)
hold on
plot(pval(1:10),ones(length(pval(1:10))),'om','Linewidth',2)
plot(sort_bg_2,2.*ones(length(sort_bg_2)),'dr','Linewidth',2)
plot(pval(11:20),2.*ones(length(pval(11:20))),'dm','Linewidth',2)
plot(sort_bg_3, 3.*ones(length(sort_bg_2)),'vg','Linewidth',2)
plot(pval(21:30),3.*ones(length(pval(21:30))),'vm','Linewidth',2)
plot([threshold_1 threshold_1],[0 8],'r','LineWidth',2)
plot([threshold_2 threshold_2],[0 8],'b','LineWidth',2)
ylim([0 4])
title('Training and Test Song Projections onto W')
xlabel('Relative song location on W')
set(gca,'Fontsize',12)

% Find accuracy 
punk_test = pval(1:10);
classical_test = pval(11:20);
rap_test = pval(21:30);

% Accuracy of Rap
true_high = 0;
for i = 1:10
    if rap_test(i)>threshold_2
        true_high = true_high + 1;
    end
end
accuracy_rap = true_high/length(rap_test);

% Accuracy of Punk
true_mid = 0;
for i = 1:10
    if punk_test(i)>threshold_1 && punk_test(i)<threshold_2 
        true_mid = true_mid + 1;
    end
end
accuracy_punk = true_mid/length(punk_test);

% Accuracy of Classical
true_low = 0;
for i = 1:10
    if classical_test(i)<threshold_1
        true_low = true_low + 1;
    end
end
accuracy_classical = true_low/length(classical_test);

accuracy_total = (accuracy_rap + accuracy_punk + accuracy_classical)/3;

%% Functions
% Audio data matrix
% bg_name = band/genre name (flume), num_song = number of songs (30),
% ds = downsample number (4) 
function [bg_data] = song_compiler(bg_name, num_song, ds) % bg_name 
cell = {};
for n = 1:num_song
    file_name = '%s_%d.wav';
    file_name = sprintf(file_name, bg_name, n);
    [y, ~]= audioread(file_name);
    y = downsample(y, ds);
    cell{n} = y(:, 1); % only use the channel one inputs
end
bg_data = [cell{:}]'; % each row is a song
end

% Spectrogram
% a = 50, tau = 0.1, L = music duration, n = number of datapoints
function [sg] = spectro(data, width, tau, L, n, num_song, t) 
    a = width; % width
    tslide=0:tau:L; 
    m_gt_g = zeros(length(tslide),floor(n)); % store filtered frequency data
    sg = []; % store spectrogram of all 30 songs a row vectors
    for song = 1:num_song 
        for j=1:length(tslide)
            % Gabor filter function / window
            gabor = exp(-a*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
            m_gt = gabor.*data(song,:); % apply filter to signal (mutiplication in time domain) 
            m_gt_f = fft(m_gt); 
            m_gt_g(j,:) = fftshift(abs(m_gt_f)); % We don't want to scale it
        end

        row_vec = [];
        for i = 1:length(tslide)
            row_vec = [row_vec m_gt_g(i, :)];
        end
        sg =[sg; row_vec];
    end
    sg = sg';
end

% Trainer 
% bg_sg_n = band/genre spectrogram data, feature = # of principal components 

function [U,S,V,w,threshold_1,threshold_2,sort_bg_1,sort_bg_2,sort_bg_3,sorted_high,sorted_mid,sorted_low] = trainer(bg_sg_1,bg_sg_2,bg_sg_3,feature)
    n1 = size(bg_sg_1,2); % number of columns in each spectrogram data set 
    n2 = size(bg_sg_2,2);
    n3 = size(bg_sg_3,2);

    % SVD Decomposition
    % data matrix = n x (3 * num_song)
    [U,S,V] = svd([bg_sg_1, bg_sg_2, bg_sg_3],'econ'); 
    % U = 2811375x90
    % S = 90x90
    % V = 90x90

    % LDA
    bg_proj = S*V'; % projection onto principal components
    U = U(:,1:feature);
    % U = 2811375xfeature
    bg_proj_1 = bg_proj(1:feature,1:n1); 
    bg_proj_2 = bg_proj(1:feature,n1+1:n1+n2);
    bg_proj_3 = bg_proj(1:feature,n1+n2+1:n1+n2+n3);

    m1 = mean(bg_proj_1,2); % (10x1) mean of all columns for each row
    m2 = mean(bg_proj_2,2);
    m3 = mean(bg_proj_3,2);

    Sw = 0; % within class variances
    for k=1:n1 %(30)
        Sw = Sw + (bg_proj_1(:,k)-m1)*(bg_proj_1(:,k)-m1)'; % sigma * sigma = variance 
    end
    for k=1:n2
        Sw = Sw + (bg_proj_2(:,k)-m2)*(bg_proj_2(:,k)-m2)';
    end
    for k=1:n3
        Sw = Sw + (bg_proj_3(:,k)-m3)*(bg_proj_3(:,k)-m3)';
    end

    num_class = 3;
%     m_total = mean([bg_proj_1, bg_proj_2, bg_proj_3]);
    m_total = (m1+m2+m3)/3;
    Sb1 = (m1-m_total)*(m1-m_total)'; % between class 
    Sb2 = (m2-m_total)*(m2-m_total)'; % between class 
    Sb3 = (m3-m_total)*(m3-m_total)'; % between class 
    Sb = (Sb1+Sb2+Sb3)/num_class;

    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); % maximum eigenvalue and its associated eigenvector 
    w = w/norm(w,2);

    v_proj_1 = w'*bg_proj_1; 
    v_proj_2 = w'*bg_proj_2;  
    v_proj_3 = w'*bg_proj_3;

    sort_bg_1 = sort(v_proj_1);
    sort_bg_2 = sort(v_proj_2);
    sort_bg_3 = sort(v_proj_3);
    
    sort_mean_1 = mean(sort_bg_1);
    sort_mean_2 = mean(sort_bg_2);
    sort_mean_3 = mean(sort_bg_3);
 
    [~, sort_mean_ind] = sort([sort_mean_1, sort_mean_2, sort_mean_3]);
    sorted_high_ind = sort_mean_ind(3);
    sorted_mid_ind = sort_mean_ind(2);
    sorted_low_ind = sort_mean_ind(1);
    
    sort_bg = [sort_bg_1; sort_bg_2; sort_bg_3];
    sorted_high = sort_bg(sorted_high_ind,:);
    sorted_mid = sort_bg(sorted_mid_ind,:);
    sorted_low = sort_bg(sorted_low_ind,:);
  
    t1 = length(sorted_low);
    t2 = 1;
    while sorted_low(t1)>sorted_mid(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold_1 = (sorted_low(t1)+sorted_mid(t2))/2;

    t2 = length(sorted_mid);
    t3 = 1;
    while sorted_mid(t2)>sorted_high(t3)
        t2 = t2-1;
        t3 = t3+1;
    end
    threshold_2 = (sorted_mid(t2)+sorted_high(t3))/2;
end