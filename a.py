import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as rPng
import scipy.signal


# def correlacaoX(img, filtro):
#     num_rows, num_cols = img.shape
#     f_num_rows, f_num_cols = filtro.shape   # f_num_rows=a+1 e f_num_cols=b+1 (a e b na fórmula acima)

#     half_r_size = f_num_rows//2        # O operador // retorna a parte inteira da divisão
#     half_c_size = f_num_cols//2

#     img_padded = np.zeros((num_rows+f_num_rows-1, num_cols+f_num_cols-1), dtype=img.dtype)
#     # print('15')
#     for row in range(num_rows):
#         for col in range(num_cols):   
#             img_padded[row+half_r_size, col+half_c_size] = img[row, col]

    
#     img_filtered = np.zeros((num_rows, num_cols))
#     # print('22')

#     for row in range(num_rows):
#         # print(f'25 {num_cols}*{f_num_rows}*{f_num_cols} = {num_cols*f_num_rows*f_num_cols}')
#         for col in range(num_cols):
#             sum_region = 0
#             for s in range(f_num_rows):
#                 for t in range(f_num_cols):
#                     sum_region += filtro[s, t]*img_padded[row+s, col+t]
#             img_filtered[row, col] = sum_region
#         # print(row)
            
#     return img_filtered


def quadDiff(img, obj):
    '''Calcula a diferença quadrática entre as imagens img e obj utilizando correlação-cruzada.'''
    
    w = np.ones(obj.shape)
    img = img.astype(float)
    obj = obj.astype(float)
    print(f'chegou aqui {img.shape} and {obj.shape}')
    # imgOw = correlacaoX(img**2, w)
    imgOw = scipy.signal.correlate(img**2, w, mode='same')
    # imgOobj = correlacaoX(img, obj)
    imgOobj = scipy.signal.correlate(img, obj, mode='same')
    
    # imgOw = imgOw.astype(int)
    # imgOobj = imgOobj.astype(int)
    imgQuad = np.sum(obj**2)
    # plt.matshow(imgQuad)
    # plt.show()
    # imgQuad = imgOobj.astype(int)
    img_diff = imgOw + imgQuad 
    # plt.matshow(img_diff)
    img_diff -= 2*imgOobj
    # plt.matshow(img_diff)
    # plt.show()

    # plt.matshow(img_diff)
    # plt.matshow(imgOw)
    # # plt.matshow(np.sum(obj**2))
    # plt.matshow(- 2*imgOobj)
    # plt.show()

    # print('tres')

    return img_diff

def minimalDiff(diffQuads):
    result = []
    idxDiff = 0
    minVal = np.Infinity
    for img in diffQuads:
        rows, cols = img.shape
        pos = -1
        for i in range(rows):
            for j in range(cols):
                if img[i,j] < minVal:
                    minVal = img[i,j]
                    pos = (i,j)
                    result = (minVal,pos,idxDiff)
        print(idxDiff, minVal)
        idxDiff += 1   
    return result
    
def downsampleColor(img):
    '''Gera uma nova imagem com metade do tamanho da imagem de entrada. A imagem de
       entrada é suavizada utilizando um filtro gaussiano e amostrada a cada 2 pixels'''
    # Filtro gaussiano
    filtro = np.array([[1,  4,  6,  4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]])
    filtro = filtro/256.
    
    img = img.astype(float)
    num_rows, num_cols, canais = img.shape
    half_num_rows = (num_rows+1)//2
    half_num_cols = (num_cols+1)//2
    
    img_smooth = np.array([scipy.signal.convolve(img[:,:,0], filtro, mode = 'same'),scipy.signal.convolve(img[:,:,1], filtro, mode = 'same'),scipy.signal.convolve(img[:,:,2], filtro, mode = 'same')])
    img_down = np.zeros((half_num_rows,half_num_cols,3))
    for row in range(0, half_num_rows):
        for col in range(half_num_cols):
            img_down[row, col, 0] = img_smooth[0,  2*row, 2*col]
            img_down[row, col, 1] = img_smooth[1, 2*row, 2*col]
            img_down[row, col, 2] = img_smooth[2, 2*row, 2*col]
    img_down = img_down.astype(int)
            
    return img_down


def buildPyramid(img, depth):
    pyramid = [img]
    for i in range(1,depth+1):
        print(i)
        pyramid.append(downsampleColor(pyramid[i-1]))
    
    return pyramid

def validSize(Io, Ig):
    k,l, _ = Io.shape
    r,c, _= Ig.shape
    return k <= r and l <= c 

def buildDiff(pyramid, Ig):
    differences = []
    idx = 0
    for img in pyramid:
        print(idx, img.shape)
        idx += 1
        if(validSize(img, Ig)):
            differences.append(quadDiff(Ig[:,:,0],img[:,:,0]) + quadDiff(Ig[:,:,1],img[:,:,1]) + quadDiff(Ig[:,:,2],img[:,:,2])  )
    return differences  

def finding(best, Ig, pyramid):
    newIg = Ig
    minval, pos, idx = best
    iOResized = pyramid[idx]
    sh = iOResized.shape

    newIg[:,:,0] = draw_rectangle(Ig[:,:,0], pos, sh, 255)
    newIg[:,:,1] = draw_rectangle(Ig[:,:,1], pos, sh, 0)
    newIg[:,:,2] = draw_rectangle(Ig[:,:,2], pos, sh, 0)

    return newIg
    
    

def draw_rectangle(img_g, center, size, color):
    '''Desenha um quadrado em uma cópia do array img_g. center indica o centro do quadrado
       e size o tamanho.'''
    
    half_num_rows_obj = size[0]//2
    half_num_cols_obj = size[1]//2

    img_rectangle = img_g.copy()
    pt1 = (center[1]-half_num_cols_obj, center[0]-half_num_rows_obj)
    pt2 = (center[1]+half_num_cols_obj, center[0]+half_num_rows_obj)
    cv2.rectangle(img_rectangle, pt1=pt1, pt2=pt2, color=color, thickness=5)
    
    return img_rectangle


    
    

Ig = rPng.imread('c.jpg')
Io = rPng.imread('d.jpg')
# plt.matshow(Io[:,:,0])
# plt.matshow(Io[:,:,1])
# plt.matshow(Io[:,:,2])
# plt.matshow(Ig[:,:,0])
# plt.matshow(Ig[:,:,1])
# plt.matshow(Ig[:,:,2])
# plt.matshow(Ig)
plt.show()
row, col, canais = Io.shape
while True:
    n = 4#int(math.log2(min(row, col)))#int(input())
    if 2**n <= min(row, col):
        break
    print(f'TU É BURRO É? QUER DIVIDIR MAIS DO QUE PODE? {Io.shape} and {min(row, col)}')

# bill é nossa piramide, ela armazena o retorno da função que constroi a
# piramide de uma imagem (Io), com profundidade n (vai até R/2^n), o nome
# da variável deriva da inspiração provida por uma série de animação denominada
# Gravity Falls, produzida e transmitida pela emissora Disney Channel 
print('piramix')
bill = buildPyramid(Io, n)
print('pyramid ok')

# spongeBob armazena as matrizes de diferenças quadráticas das imagens,
# será utilizada posteriormente para determinar a melhor posição onde o objeto
# pode ter sido encontrado, o nome deriva de uma série de animação criada por 
# Stephen Hillenburg, denomidada originalmente como SpongeBob SquarePants. 
print('buildando diff')
spongeBob = buildDiff(bill, Ig)
print('diff ok')

# cr7 armazena indices para o menor valor encontrado das diferenças
# quadráticas, a posição onde ele se encontra e por fim o index referente 
# a qual matriz da diferença de quadrados refere-se essa posição, o nome deriva
# de uma celebridade, conhecida por jogar futebol, conhecida por Cristiano
# Ronaldo ou cr7 para os fãs, tal nome foi escolhido, pois a mesma em uma
# entrevista proferiu a seguinte sentença: "Eu sou o melhor", o que se
# correlaciona com a função da variável, que armazena o melhor resultado.
print('entering eu sou o milhor')
cr7 = minimalDiff(spongeBob)
print('cr7 ok')

# aqui alteramos a última posição da tupla, para nos retornar a qual nível da piramide
# se refere essa diferença quadrática, será utilizada para delimitar a área da imagem,
# onde está o objeto que desejamos encontrar
cr7 = (cr7[0],cr7[1],cr7[2] + len(bill) - len(spongeBob))

# nemo armazena a imagem delimitada do melhor valor encontrado pelas diferenças
# quadráticas, em sumo, dado o indice, busca na imagem original, tal indice e
# exibe a imagem encontrada delimitada, o nome deriva de um longa metragem de
# animação produzida e transmitida pela emissora Disney Channel, denominada
# originalmente Finding nemo
nemo = finding(cr7,Ig,bill)
print(cr7)

# for i in bill:
    # plt.matshow(i)
# for i in spongeBob:
#     plt.matshow(i)
#     # plt.figure(figsize=[7,7])
#     # plt.imshow(i)
plt.figure(figsize=[7,7])
plt.imshow(nemo)
# plt.figure(figsize=[7,7])
# plt.imshow(Ig)
# plt.figure(figsize=[7,7])
# plt.imshow(nemo)
plt.show()