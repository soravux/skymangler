import os
import sys

listeCmds = ['python test_AE.py 5000 500 50 --outputprefix=t1 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 5000 1000 200 40 --outputprefix=t2 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 5000 2000 800 300 80 15 --outputprefix=t3 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 200 15 --outputprefix=t4 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t5 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 200 15 --outputprefix=t6 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t7 --lrate=0.2 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t8 --lrate=0.01 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t9 --lrate=0.1 --bruit=0.45 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t10 --lrate=0.1 --bruit=0.05 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t11 --batchsize=50 --lrate=0.1 --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t12 --lrate=0.1 --noisetype=gaussian --bruit=0.25 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t13 --lrate=0.1 --noisetype=gaussian --bruit=0.15 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                'python test_AE.py 2000 500 200 50 15 --outputprefix=t14 --lrate=0.1 --noisetype=gaussian --bruit=0.45 --datasetpath=../data/ --datasetname="envmap.exr.mat" --datasettype=MAT --datasetlist=listeFilesMat --datasetpcttrain=0.7 --nan=remove',
                ] 

if __name__ == '__main__':
    i = int(sys.argv[1])
    os.system(listeCmds[i]-1)