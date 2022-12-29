q = 0.02;
p = 0.98;

% load image
lena = imread('lena_std.tif');
cabai = imread("cabai1.jpg");

% load watermark
wm = imread("wm2.bmp");
wm = uint8(imcomplement(wm) * 255);

% embedding watermark and extracting watermark
b_size = 256;
w_cabai = blockproc(cabai, [b_size b_size], @(x) embbed(x.data, wm, p, q));
wm_cabai_ext = blockproc(w_cabai, [b_size b_size], @(x) extract(cabai, x, p, q));

% drawing
figure(4);
subplot(2, 2, 1); imshow(cabai, [0 255]); title("Citra yang akan Diberikan Watermark");
subplot(2, 2, 2); imshow(w_cabai, [0 255]); title("Citra Berwatermark");
subplot(2, 2, 3); imshow(wm, [0 255]); title("Watermark yang akan disisipkan");
subplot(2, 2, 4); imshow(wm_cabai_ext(1:256, 1:256), [0 255]); title("Watermark yang Terekstrak");

% embedding watermark and extracting watermark
b_size = 512;
w_lena = blockproc(lena, [b_size b_size], @(x) embbed(x.data, wm, p, q));
wm_ext = blockproc(w_lena, [b_size b_size], @(x) extract(lena, x, p, q));

% drawing
figure(1);
subplot(2, 2, 1); imshow(lena, [0 255]); title("Citra yang akan Diberikan Watermark");
subplot(2, 2, 2); imshow(w_lena, [0 255]); title("Citra Berwatermark");
subplot(2, 2, 3); imshow(wm, [0 255]); title("Watermark yang akan disisipkan");
subplot(2, 2, 4); imshow(wm_ext(1:b_size, 1:b_size), [0 255]); title("Watermark yang Terekstrak");

% writing watermarked images
imwrite(w_lena, "lena_watermarked.tif");
imwrite(w_cabai, "cabai_watermarked.jpg");

% attack resize ke 256x256
w_lena_256 = imresize(w_lena, [256 256]);
imwrite(w_lena_256, "lena_watermarked_256.tif");
wm_ext_256 = blockproc(imresize(w_lena_256, size(lena, [1 2])), [b_size b_size], @(x) extract(lena, x, p, q));

% attack resize ke 1024x1024
w_lena_1024 = imresize(w_lena, [1024 1024]);
imwrite(w_lena_1024, "lena_watermarked_256.tif");
wm_ext_1024 = blockproc(imresize(w_lena_1024, size(lena, [1 2])), [b_size b_size], @(x) extract(lena, x, p, q));

% attack noise salt n pepper
w_lena_snp = imnoise(w_lena, "salt & pepper", 0.01);
imwrite(w_lena_snp, "lena_watermarked_snp_512.tif");
[w_lena_snp_r, w_lena_snp_g, w_lena_snp_b] = imsplit(w_lena_snp);
w_lena_snp_r = medfilt2(w_lena_snp_r);
w_lena_snp_g = medfilt2(w_lena_snp_g);
w_lena_snp_b = medfilt2(w_lena_snp_b);
wm_ext_snp_512 = blockproc(cat(3, w_lena_snp_r, w_lena_snp_g, w_lena_snp_b), [b_size b_size], @(x) extract(lena, x, p, q));

% attack crop
w_lena_crop_400 = padarray(imcrop(w_lena, [0 0 400 400]), [112 112], 0, "post");
imwrite(w_lena_crop_400, "lena_watermarked_crop_400.tif");
wm_ext_crop_400 = blockproc(w_lena_crop_400, [b_size b_size], @(x) extract(lena, x, p, q));

% attack noise gaussian
w_lena_gaussian = imnoise(w_lena, "gaussian", 0, 0.005);
imwrite(w_lena_gaussian, "lena_watermarked_gaussian_512.tif");
wm_ext_gaussian_512 = blockproc(w_lena_gaussian, [b_size b_size], @(x) extract(lena, x, p, q));

imwrite(w_lena, "lena_compressed.jpg");
w_lena_compressed = imread("lena_compressed.jpg");
wm_ext_compressed = blockproc(w_lena_compressed, [b_size b_size], @(x) extract(lena, x, p, q));

% drawing
figure(2);
subplot(2, 3, 1); imshow(w_lena_256, [0 255]); title({"Citra Watermarked", "Resized (256x256)"});
subplot(2, 3, 2); imshow(w_lena_1024, [0 255]); title({"Citra Watermarked", "Resized (1024x1024)"});
subplot(2, 3, 3); imshow(w_lena_snp, [0 255]); title({"Citra Watermarked", "dengan Noise Salt & Pepper"});
subplot(2, 3, 4); imshow(w_lena_crop_400, [0 255]); title({"Citra Watermarked", "yang Di-Crop"});
subplot(2, 3, 5); imshow(w_lena_gaussian, [0 255]); title({"Citra Watermarked", "dengan Noise Gaussian"});
subplot(2, 3, 6); imshow(w_lena_compressed, [0 255]); title({"Citra Watermarked", "yang Dikompresi"});

% drawing
figure(3);
subplot(2, 3, 1); imshow(wm_ext_256, [0 255]); title({"Watermark yang Terekstrak", "dari Citra Resized (256x256)"});
subplot(2, 3, 2); imshow(wm_ext_1024, [0 255]); title({"Watermark yang Terekstrak", "dari Citra Resized (1024x1024)"});
subplot(2, 3, 3); imshow(wm_ext_snp_512, [0 255]); title({"Watermark yang Terekstrak", "dari Citra dengan Noise Salt & Pepper"});
subplot(2, 3, 4); imshow(wm_ext_crop_400, [0 255]); title({"Watermark yang Terekstrak", "dari Citra yang Di-Crop"});
subplot(2, 3, 5); imshow(wm_ext_gaussian_512, [0 255]); title({"Watermark yang Terekstrak", "dari Citra dengan Noise Gaussian"});
subplot(2, 3, 6); imshow(wm_ext_compressed, [0 255]); title({"Watermark yang Terekstrak", "dari Citra yang Dikompresi"});

disp("PSNR lena");
disp(psnr(lena, w_lena));

disp("PSNR cabai");
disp(psnr(cabai, w_cabai));

function ext = extract(origin, bs, p, q)
    wmi = bs.data;
    [wmi_r, wmi_g, wmi_b] = imsplit(wmi);

    startInd = bs.location;
    endInd   = bs.location+bs.blockSize-1;
    o = origin(startInd(1):endInd(1), startInd(2):endInd(2), 1:3);
    [origin_r, origin_g, origin_b] = imsplit(o);

    ext_r = extract_dwt(origin_r, wmi_r, p, q);
    ext_g = extract_dwt(origin_g, wmi_g, p, q);
    ext_b = extract_dwt(origin_b, wmi_b, p, q);

    ext = uint8((ext_r + ext_g + ext_b) / 3);
end

function w_img = embbed(origin, wm, p, q)
    [origin_r, origin_g, origin_b] = imsplit(origin);
    
    w_img_r = embbed_dwt(origin_r, wm, p, q);
    w_img_g = embbed_dwt(origin_g, wm, p, q);
    w_img_b = embbed_dwt(origin_b, wm, p, q);
    w_img = cat(3, w_img_r, w_img_g, w_img_b);

    w_img = uint8(w_img);
end

function img = embbed_dwt(origin, wm, p, q)
    [LL,LH,HL,HH] = dwt2(origin, 'haar');
    [LL1,LH1,HL1,HH1] = dwt2(LL, 'haar');
    [LL2,LH2,HL2,HH2] = dwt2(LL1, 'haar');

    [LL_w,LH_w,HL_w,HH_w] = dwt2(imresize(wm, size(origin)),'haar');
    [LL1_w,LH1_w,HL1_w,HH1_w] = dwt2(LL_w,'haar');
    [LL2_w,LH2_w,HL2_w,HH2_w] = dwt2(LL1_w, 'haar');

    img_level_0 = p * LL2 + q * LL2_w;
    img_level_1 = idwt2(img_level_0, LH2, HL2, HH2, 'haar');
    img_level_2 = idwt2(img_level_1, LH1, HL1, HH1, 'haar');
    img = idwt2(img_level_2, LH, HL, HH, 'haar');
end

function img = extract_dwt(origin, watermarked, p, q)
    [LL_w,LH_w,HL_w,HH_w] = dwt2(watermarked, 'haar');
    [LL1_w,LH1_w,HL1_w,HH1_w] = dwt2(LL_w, 'haar');
    [LL2_w,LH2_w,HL2_w,HH2_w] = dwt2(LL1_w, 'haar');

    [LL_o,LH,HL,HH] = dwt2(origin, 'haar');
    [LL1_o,LH1,HL1,HH1] = dwt2(LL_o, 'haar');
    [LL2_o,LH2,HL2,HH2] = dwt2(LL1_o, 'haar');

    img_level_0 = (LL2_w - p * LL2_o) / q;
    img = idwt2(img_level_0, [], [], [], 'haar');
    img = idwt2(img, [], [], [], 'haar');
    img = idwt2(img, [], [], [], 'haar');
end