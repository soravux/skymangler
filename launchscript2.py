import os
import sys

listeCmds = ['python test_AE.py 2000 500 200 50 15 --outputprefix=t15 --lrate=0.3 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t16 --lrate=0.1 --bruit=0.65 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t17 --lrate=0.1 --bruit=0.25 --disttype=abs --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t18 --lrate=0.01 --bruit=0.25 --disttype=abs --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t19 --lrate=0.1 --bruit=0.25 --disttype=eucli --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t20 --lrate=0.1 --bruit=0.05 --disttype=eucli --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                ] 

if __name__ == '__main__':
    i = int(sys.argv[1])
    os.system(listeCmds[i-1])