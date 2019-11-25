import numpy as np

### loading the data
#
def loadData(path='./data48x48.npz', forCNN=False):

    rv = np.load(path)
    XL = rv['XL'].astype(np.float32)
    XT = rv['XT'].astype(np.float32)
    Xmean = rv['Xmean'].astype(np.float32)

    if forCNN:
        XL = XL.reshape((-1, 48, 48, 3))
        XT = XT.reshape((-1, 48, 48, 3))
        Xmean = Xmean.reshape((48, 48, 3))

    print(XL.shape, XT.shape, Xmean.shape)

    return XL, XT, Xmean


### (batchsize, H, W, C) => (batchsize, C, H, W)
#
def HWC2CHW(X):

    return X.transpose((0, 3, 1, 2))


### (batchsize, C, H, W) => (batchsize, H, W, C) 
#
def CHW2HWC(X):

    return X.transpose((0, 2, 3, 1))


### making mini batch indicies
#
def makeBatchIndex(N, batchsize):

    idx = np.random.permutation(N)
        
    nbatch = int(np.ceil(float(N) / batchsize))
    idxB = np.zeros(( nbatch, N ), dtype = bool)
    for ib in range(nbatch - 1):
        idxB[ib, idx[ib*batchsize:(ib+1)*batchsize]] = True
    ib = nbatch - 1
    idxB[ib, idx[ib*batchsize:]] = True

    return idxB


### making a montage of selected images
#
def montage(Z, Xmean, idx):
    
    ny, nx = idx.shape

    ZZ = Z[idx.reshape(-1), :] + Xmean
    ZZ *= 255
    
    gap = 4         #  画像間のスペース
    ncol = nrow = 48

    # 並べた画像の幅と高さ
    width  = nx * (ncol + gap) + gap
    height = ny * (nrow + gap) + gap

    # 画像の作成
    mon = np.zeros((height, width, 3), dtype = int) + 128
    for iy in range(ny):
        lty = iy*(nrow + gap) + gap
        for ix in range(nx):
            ltx = ix*(ncol + gap) + gap
            mon[lty:lty+nrow, ltx:ltx+ncol] = ZZ[iy*nx+ix, :].reshape((nrow, ncol, 3))
            
    return mon