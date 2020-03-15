%% AMATH 482 Homework 3
% Seong Hyun Han
% 2/20/20

%% Test 1 - Video Processing 
%% Cam1_1

%Starter code - load video and store data in matrix X
clear all; close all; clc
load('cam1_1.mat')
% implay(vidFrames1_1) play each fram

[y_res, x_res, rgb, numFrames] = size(vidFrames1_1);
position = [];
for j = 1:numFrames
X = vidFrames1_1(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket
[y_len,x_len,~] = size(gray_X);
crop = zeros(y_len,x_len);
c_start = 290;
c_end = 390;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = gray_X(1:y_len, 1:x_len).*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.995); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[55;25]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x1_loc = [];
y1_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x1_loc = [x1_loc; pos_cell{1,i}(1,1)];
    y1_loc(i, 1) = y1_loc(i, 1) - pos_cell{1,i}(1,2);
end

figure(1)
subplot(1,3,1)
plot(x1_loc, y1_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 1 Camera 1');

% Cam2_1
load('cam2_1.mat')
% implay(vidFrames2_1)

[y_res, x_res, rgb, numFrames] = size(vidFrames2_1);
position = [];
for j = 1:numFrames
X = vidFrames2_1(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket
[y_len,x_len,~] = size(gray_X);
crop = zeros(y_len,x_len);
c_start = 230;
c_end = 380;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = gray_X(1:y_len, 1:x_len).*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.97); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[60;30]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x2_loc = [];
y2_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x2_loc = [x2_loc; pos_cell{1,i}(1,1)];
    y2_loc(i, 1) = y2_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,2)
plot(x2_loc, y2_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10)
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 1 Camera 2');

% Cam3_1

load('cam3_1.mat')
% implay(vidFrames3_1)

[y_res, x_res, rgb, numFrames] = size(vidFrames3_1);
position = [];
for j = 1:numFrames
X = vidFrames3_1(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 230;
ch_end = 345;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 250;
c_end = 500;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.90); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[30;60]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x3_loc = [];
y3_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x3_loc = [x3_loc; pos_cell{1,i}(1,1)];
    y3_loc(i, 1) = y3_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,3)
plot(x3_loc, y3_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 1 Camera 3');

%% Test 1 - SVD
% make all of the camera videos in sync 
[~,y1_min] = min(y1_loc(1:50));
[~,y2_min] = min(y2_loc(1:50));
[~,x3_max] = max(x3_loc(1:50));

x1_loc = x1_loc(y1_min:end);
y1_loc = y1_loc(y1_min:end);

x2_loc = x2_loc(y2_min:end);
y2_loc = y2_loc(y2_min:end);

x3_loc = x3_loc(x3_max:end);
y3_loc = y3_loc(x3_max:end);

% find an averaged x and y coordinate system for each camera 
% to reduce the effect of the svd giving more weight on just 
% absolute location values 

% average row values
x1_ave = mean(x1_loc);
x2_ave = mean(x2_loc);
x3_ave = mean(x3_loc);

y1_ave = mean(y1_loc);
y2_ave = mean(y2_loc);
y3_ave = mean(y3_loc);

% find the minimum matrix size
min_size = min(length(x1_loc), length(x2_loc));
min_size = min(min_size, length(x3_loc));

% reduces matrix size to minimum matrix size
x1 = x1_loc(1:min_size);
x2 = x2_loc(1:min_size);
x3 = x3_loc(1:min_size);

y1 = y1_loc(1:min_size);
y2 = y2_loc(1:min_size);
y3 = y3_loc(1:min_size);

% subtract each row element by its average
for n = 1:length(x1)
    x1(n,1) = x1(n,1) - x1_ave;
    x2(n,1) = x2(n,1) - x2_ave;
    x3(n,1) = x3(n,1) - x3_ave;
    
    y1(n,1) = y1(n,1) - y1_ave;
    y2(n,1) = y2(n,1) - y2_ave;
    y3(n,1) = y3(n,1) - y3_ave;
end

% Contruct coordinates into a single matrix
X_svd = [x1';y1';x2';y2';x3';y3'];

% Take the SVD of the single matrix using "economy size" 
% decomposition - low-dimensional approximations
[U,S,V] = svd(X_svd,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(2)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',10) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 1');

% plot of all camera angles showing major variation direction (z)
subplot(2,2,1)
plot(1:length(y1),y1,1:length(y1),y2,1:length(y1),x3,'LineWidth',1)
set(gca,'Xlim',[0,length(y1)],'Ylim',[-100,100],'Fontsize',10) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Bucket Displacement in Time - Test 1');

% plot of data projections onto pricipal components  
subplot(2,2,3)
X_proj = U'*X_svd;
plot(1:length(X_proj(1,:)), X_proj(1,:),'LineWidth',1)
hold on
plot(1:length(X_proj(2,:)), X_proj(2,:),'LineWidth',1)
legend("PC 1","PC 2")
set(gca,'Xlim',[0,length(X_proj(1,:))],'Fontsize',10) 
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Data Projections onto Principal Components - Test 1');
%% Test 2 - Video Processing 

%% Cam1_2
clear all; close all; clc
load('cam1_2.mat')
% implay(vidFrames1_2)

[y_res, x_res, rgb, numFrames] = size(vidFrames1_2);
position = [];
for j = 1:numFrames
X = vidFrames1_2(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 200;
ch_end = 440;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 300;
c_end = 440;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.92); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[55;50]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x1_loc = [];
y1_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x1_loc = [x1_loc; pos_cell{1,i}(1,1)];
    y1_loc(i, 1) = y1_loc(i, 1) - pos_cell{1,i}(1,2);
end

figure(8)
subplot(1,3,1)
plot(x1_loc, y1_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 2 Camera 1');

% Cam2_2

load('cam2_2.mat')
% implay(vidFrames2_2)

[y_res, x_res, rgb, numFrames] = size(vidFrames2_2);
position = [];
for j = 1:numFrames
X = vidFrames2_2(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 60;
ch_end = 420;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 180;
c_end = 440;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.965); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[80;85]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x2_loc = [];
y2_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x2_loc = [x2_loc; pos_cell{1,i}(1,1)];
    y2_loc(i, 1) = y2_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,2)
plot(x2_loc, y2_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 2 Camera 2');

% Cam3_2

load('cam3_2.mat')
% implay(vidFrames3_2)

[y_res, x_res, rgb, numFrames] = size(vidFrames3_2);
position = [];
for j = 1:numFrames
X = vidFrames3_2(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 190;
ch_end = 346;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 286;
c_end = 495;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.90); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[30;60]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x3_loc = [];
y3_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x3_loc = [x3_loc; pos_cell{1,i}(1,1)];
    y3_loc(i, 1) = y3_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,3)
plot(x3_loc, y3_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 2 Camera 3');

%% Test 2 - SVD
% make all of the camera videos in sync 
[~,y1_min] = min(y1_loc(1:50));
[~,y2_min] = min(y2_loc(1:50));
[~,x3_max] = max(x3_loc(1:50));

x1_loc = x1_loc(y1_min:end);
y1_loc = y1_loc(y1_min:end);

x2_loc = x2_loc(y2_min:end);
y2_loc = y2_loc(y2_min:end);

x3_loc = x3_loc(x3_max:end);
y3_loc = y3_loc(x3_max:end);

% find an averaged x and y coordinate system for each camera 
% to reduce the effect of the svd giving more weight on just 
% absolute location values 

% average row values
x1_ave = mean(x1_loc);
x2_ave = mean(x2_loc);
x3_ave = mean(x3_loc);

y1_ave = mean(y1_loc);
y2_ave = mean(y2_loc);
y3_ave = mean(y3_loc);

% find the minimum matrix size
min_size = min(length(x1_loc), length(x2_loc));
min_size = min(min_size, length(x3_loc));

% reduces matrix size to minimum matrix size
x1 = x1_loc(1:min_size);
x2 = x2_loc(1:min_size);
x3 = x3_loc(1:min_size);

y1 = y1_loc(1:min_size);
y2 = y2_loc(1:min_size);
y3 = y3_loc(1:min_size);

% subtract each row element by its average
for n = 1:length(x1)
    x1(n,1) = x1(n,1) - x1_ave;
    x2(n,1) = x2(n,1) - x2_ave;
    x3(n,1) = x3(n,1) - x3_ave;
    
    y1(n,1) = y1(n,1) - y1_ave;
    y2(n,1) = y2(n,1) - y2_ave;
    y3(n,1) = y3(n,1) - y3_ave;
end

% Contruct coordinates into a single matrix
X_svd = [x1';y1';x2';y2';x3';y3'];

% Take the SVD of the single matrix using "economy size" 
% decomposition - low-dimensional approximations
[U,S,V] = svd(X_svd,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(9)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',10) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 2');

% plot of all camera angles showing major variation direction (z)
subplot(2,2,1)
plot(1:length(y1),y1,1:length(y1),y2,1:length(y1),x3,'LineWidth',1)
set(gca,'Xlim',[0,length(y1)],'Ylim',[-150,150],'Fontsize',10) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Bucket Displacement in Time - Test 2');

% plot of data projections onto pricipal components  
subplot(2,2,3)
X_proj = U'*X_svd;
plot(1:length(X_proj(1,:)), X_proj(1,:),'LineWidth',1)
hold on
plot(1:length(X_proj(2,:)), X_proj(2,:),'LineWidth',1)
plot(1:length(X_proj(3,:)), X_proj(3,:),'LineWidth',1)
legend("PC 1","PC 2","PC 3")
set(gca,'Xlim',[0,length(X_proj(1,:))],'Fontsize',10) 
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Data Projections onto Principal Components - Test 2');

%% Test 3 - Video Processing 

%% Cam1_3
clear all; close all; clc
load('cam1_3.mat')
% implay(vidFrames1_3)

[y_res, x_res, rgb, numFrames] = size(vidFrames1_3);
position = [];
for j = 1:numFrames
X = vidFrames1_3(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 233;
ch_end = 415;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 280;
c_end = 401;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.94); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[70;55]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x1_loc = [];
y1_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x1_loc = [x1_loc; pos_cell{1,i}(1,1)];
    y1_loc(i, 1) = y1_loc(i, 1) - pos_cell{1,i}(1,2);
end

figure(15)
subplot(1,3,1)
plot(x1_loc, y1_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 3 Camera 1');

% Cam2_3

load('cam2_3.mat')
% implay(vidFrames2_3)

[y_res, x_res, rgb, numFrames] = size(vidFrames2_3);
position = [];
for j = 1:numFrames
X = vidFrames2_3(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 180;
ch_end = 407;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 205;
c_end = 426;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.95); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[80;85]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x2_loc = [];
y2_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x2_loc = [x2_loc; pos_cell{1,i}(1,1)];
    y2_loc(i, 1) = y2_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,2)
plot(x2_loc, y2_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 3 Camera 2');

% Cam3_3
load('cam3_3.mat')
% implay(vidFrames3_3)

[y_res, x_res, rgb, numFrames] = size(vidFrames3_3);
position = [];
for j = 1:numFrames
X = vidFrames3_3(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 150;
ch_end = 355;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 263;
c_end = 465;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.90); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[30;60]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x3_loc = [];
y3_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x3_loc = [x3_loc; pos_cell{1,i}(1,1)];
    y3_loc(i, 1) = y3_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,3)
plot(x3_loc, y3_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 3 Camera 3');

%% Test 3 - SVD
% make all of the camera videos in sync 
[~,y1_min] = min(y1_loc(1:50));
[~,y2_min] = min(y2_loc(1:50));
[~,x3_max] = max(x3_loc(1:50));

x1_loc = x1_loc(y1_min:end);
y1_loc = y1_loc(y1_min:end);

x2_loc = x2_loc(y2_min:end);
y2_loc = y2_loc(y2_min:end);

x3_loc = x3_loc(x3_max:end);
y3_loc = y3_loc(x3_max:end);

% find an averaged x and y coordinate system for each camera 
% to reduce the effect of the svd giving more weight on just 
% absolute location values 

% average row values
x1_ave = mean(x1_loc);
x2_ave = mean(x2_loc);
x3_ave = mean(x3_loc);

y1_ave = mean(y1_loc);
y2_ave = mean(y2_loc);
y3_ave = mean(y3_loc);

% find the minimum matrix size
min_size = min(length(x1_loc), length(x2_loc));
min_size = min(min_size, length(x3_loc));

% reduces matrix size to minimum matrix size
x1 = x1_loc(1:min_size);
x2 = x2_loc(1:min_size);
x3 = x3_loc(1:min_size);

y1 = y1_loc(1:min_size);
y2 = y2_loc(1:min_size);
y3 = y3_loc(1:min_size);

% subtract each row element by its average
for n = 1:length(x1)
    x1(n,1) = x1(n,1) - x1_ave;
    x2(n,1) = x2(n,1) - x2_ave;
    x3(n,1) = x3(n,1) - x3_ave;
    
    y1(n,1) = y1(n,1) - y1_ave;
    y2(n,1) = y2(n,1) - y2_ave;
    y3(n,1) = y3(n,1) - y3_ave;
end

% Contruct coordinates into a single matrix
X_svd = [x1';y1';x2';y2';x3';y3'];

% Take the SVD of the single matrix using "economy size" 
% decomposition - low-dimensional approximations
[U,S,V] = svd(X_svd,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(16)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',10) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 3');

% plot of all camera angles showing major variation direction (z)
subplot(2,2,1)
plot(1:length(y1),y1,1:length(y1),y2,1:length(y1),x3,'LineWidth',1)
set(gca,'Xlim',[0,length(y1)],'Ylim',[-100,100],'Fontsize',10) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Bucket Displacement in Time - Test 3');

% plot of data projections onto pricipal components  
subplot(2,2,3)
X_proj = U'*X_svd;
plot(1:length(X_proj(1,:)), X_proj(1,:),'LineWidth',1)
hold on
plot(1:length(X_proj(2,:)), X_proj(2,:),'LineWidth',1)
plot(1:length(X_proj(3,:)), X_proj(3,:),'LineWidth',1)
plot(1:length(X_proj(4,:)), X_proj(4,:),'LineWidth',1)
legend("PC 1","PC 2","PC 3","PC 4")
set(gca,'Xlim',[0,length(X_proj(1,:))],'Fontsize',10) 
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Data Projections onto Principal Components - Test 3');


%% Test 4 - Video Processing 

%% Cam1_4
clear all; close all; clc
load('cam1_4.mat')
% implay(vidFrames1_4)

[y_res, x_res, rgb, numFrames] = size(vidFrames1_4);
position = [];
for j = 1:numFrames
X = vidFrames1_4(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 216;
ch_end = 420;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 301;
c_end = 469;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.95); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[70;55]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x1_loc = [];
y1_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x1_loc = [x1_loc; pos_cell{1,i}(1,1)];
    y1_loc(i, 1) = y1_loc(i, 1) - pos_cell{1,i}(1,2);
end

figure(21)
subplot(1,3,1)
plot(x1_loc, y1_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 4 Camera 1');

% Cam2_4

load('cam2_4.mat')
% implay(vidFrames2_4)

[y_res, x_res, rgb, numFrames] = size(vidFrames2_4);
position = [];
for j = 1:numFrames
X = vidFrames2_4(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 83;
ch_end = 365;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 210;
c_end = 434;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.97); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[80;50]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x2_loc = [];
y2_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x2_loc = [x2_loc; pos_cell{1,i}(1,1)];
    y2_loc(i, 1) = y2_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,2)
plot(x2_loc, y2_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 4 Camera 2');

% Cam3_4 
load('cam3_4.mat')
% implay(vidFrames3_4)

[y_res, x_res, rgb, numFrames] = size(vidFrames3_4);
position = [];
for j = 1:numFrames
X = vidFrames3_4(:,:,:,j); % read each frame
gray_X = rgb2gray(X); % change to gray scale

% crop gray scale image to extract only the bucket (horizontal)
[y_len,x_len,~] = size(gray_X);
crop_h = zeros(y_len,x_len);
ch_start = 130;
ch_end = 295;
ch_width = (ch_end - ch_start) + 1; 
crop_h(ch_start:ch_end, 1:x_len) = ones(ch_width, x_len);
cgh_X = gray_X(1:y_len, 1:x_len).*uint8(crop_h);

% crop gray scale image to extract only the bucket (vertical)
crop = zeros(y_len,x_len);
c_start = 280;
c_end = 522;
c_width = (c_end - c_start) + 1; 
crop(1:y_len, c_start:c_end) = ones(y_len, c_width);
cg_X = cgh_X.*uint8(crop);

% extract only the bucket/light based on  
BW = imbinarize(cg_X, 0.905); % binarize grayscale image by thresholding
BW_filt = medfilt2(BW,'symmetric'); % 2D median filtering 
se = strel('rectangle',[50;60]); % create morphological structuring element 
BW_fd = imdilate(BW_filt, se); % performs dilation using the structuring element 

% find connected components in binary image and position
CC = bwconncomp(BW_fd);
S = regionprops(CC, 'Centroid');
position = [position; S];

% imshow(BW_fd);
% drawnow
end

% convert structure array to cell array
pos_cell = struct2cell(position);
x3_loc = [];
y3_loc = y_res.*ones(length(pos_cell),1);
for i = 1:length(pos_cell)
    x3_loc = [x3_loc; pos_cell{1,i}(1,1)];
    y3_loc(i, 1) = y3_loc(i, 1) - pos_cell{1,i}(1,2);
end

subplot(1,3,3)
plot(x3_loc, y3_loc,'.','LineWidth',9)
set(gca,'Xlim',[0,x_res],'Ylim',[0,y_res],'Fontsize',10) 
xticks(0:50:x_res)
xlabel('x (px)');
ylabel('y (px)');
title('Bucket Positions - Test 4 Camera 3');

%% Test 4 - SVD
% make all of the camera videos in sync 
[~,y1_min] = min(y1_loc(1:50));
[~,y2_min] = min(y2_loc(1:50));
[~,x3_max] = max(x3_loc(1:50));

x1_loc = x1_loc(y1_min:end);
y1_loc = y1_loc(y1_min:end);

x2_loc = x2_loc(y2_min:end);
y2_loc = y2_loc(y2_min:end);

x3_loc = x3_loc(x3_max:end);
y3_loc = y3_loc(x3_max:end);

% find an averaged x and y coordinate system for each camera 
% to reduce the effect of the svd giving more weight on just 
% absolute location values 

% average row values
x1_ave = mean(x1_loc);
x2_ave = mean(x2_loc);
x3_ave = mean(x3_loc);

y1_ave = mean(y1_loc);
y2_ave = mean(y2_loc);
y3_ave = mean(y3_loc);

% find the minimum matrix size
min_size = min(length(x1_loc), length(x2_loc));
min_size = min(min_size, length(x3_loc));

% reduces matrix size to minimum matrix size
x1 = x1_loc(1:min_size);
x2 = x2_loc(1:min_size);
x3 = x3_loc(1:min_size);

y1 = y1_loc(1:min_size);
y2 = y2_loc(1:min_size);
y3 = y3_loc(1:min_size);

% subtract each row element by its average
for n = 1:length(x1)
    x1(n,1) = x1(n,1) - x1_ave;
    x2(n,1) = x2(n,1) - x2_ave;
    x3(n,1) = x3(n,1) - x3_ave;
    
    y1(n,1) = y1(n,1) - y1_ave;
    y2(n,1) = y2(n,1) - y2_ave;
    y3(n,1) = y3(n,1) - y3_ave;
end

% Contruct coordinates into a single matrix
X_svd = [x1';y1';x2';y2';x3';y3'];

% Take the SVD of the single matrix using "economy size" 
% decomposition - low-dimensional approximations
[U,S,V] = svd(X_svd,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(16)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',10) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 4');

% plot of all camera angles showing major variation direction (z)
subplot(2,2,1)
plot(1:length(y1),y1,1:length(y1),y2,1:length(y1),x3,'LineWidth',1)
set(gca,'Xlim',[0,length(y1)],'Ylim',[-100,100],'Fontsize',10) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Bucket Displacement in Time - Test 4');

% plot of data projections onto pricipal components  
subplot(2,2,3)
X_proj = U'*X_svd;
plot(1:length(X_proj(1,:)), X_proj(1,:),'LineWidth',1)
hold on
plot(1:length(X_proj(2,:)), X_proj(2,:),'LineWidth',1)
plot(1:length(X_proj(3,:)), X_proj(3,:),'LineWidth',1)
legend("PC 1","PC 2","PC 3")
set(gca,'Xlim',[0,length(X_proj(1,:))],'Fontsize',10) 
xlabel('Time (frames)');
ylabel('Displacement (px)');
title('Data Projections onto Principal Components - Test 4');
