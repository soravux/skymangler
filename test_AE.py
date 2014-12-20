"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import os, os.path
import sys
import time


import numpy
import scipy.misc, scipy.io
import cPickle as pickle
from glob import glob

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

import argparse



class DatasetImporter:
    """
    Permet d'importer des datasets de skymaps sous
    differentes formes.
    """

    def __init__(self, dataset_type, folder, name, nanbehavior='zeros', cropWidth=0, savepkl=None, patchHidden=False):
        """
        dataset_type = MAT | EXR | PKL
        """
        self.type = dataset_type
        self.path = folder
        self.name = name
        self.cropW = cropWidth
        self.nan = nanbehavior
        self.savepkl = savepkl
        self.D, self.N = None, None
        self.patch = patchHidden

        self.dataRGB = []
        self.dataGray = []
        self.dataTheano = None

    def _rgb2gray(self, rgb):
        # Ponderation un peu arbitraire
        d = rgb[:,:,0] * 0.25 + rgb[:,:,1] * 0.25 + rgb[:,:,2] * 0.5
        return d/d[~numpy.isnan(d)].max()

    def _removeNaNs(self, gray):
        if self.nan == 'zeros':
            gray[numpy.isnan(gray)] = 0.
        elif self.nan == 'remove':
            gray[~numpy.isnan(gray)] = gray[~numpy.isnan(gray)] / gray[~numpy.isnan(gray)].max()
            gray = gray[~numpy.isnan(gray)]
        return gray

    def _crop(self, gray):
        if self.cropW == 0:
            return gray
        if len(gray.shape) == 1:
            c = int(gray.shape[0]**0.5)
            gray = gray.reshape(c, c)
        return gray[self.cropW:-self.cropW, self.cropW:-self.cropW]

    def _patch(self, gray):
        if self.patch:
            c = int(gray.shape[0]**0.5)
            gray = gray.reshape(c, c)
            gray[39:49, 103:113] = 0.
            #print(gray[41:47, 105:111])
            #from matplotlib import pyplot
            #pyplot.imshow(gray)
            #pyplot.show()
        gray = numpy.ravel(gray)
        #print(gray.shape, self.patch)
        return gray

    def _loadmat(self):
        fichiers = os.listdir(self.path)
        for fname in fichiers:
            if not self.name in fname:
                continue

            img = scipy.io.loadmat(os.path.join(self.path, fname))['envmap']
            self.dataRGB.append(img)

            # Conversion 
            gray = self._rgb2gray(img)
            # Retrait d'eventuels NaNs
            gray_noNaNs = self._removeNaNs(gray)

            gray_crop = self._crop(gray_noNaNs)

            self.dataGray.append(gray_crop)

            if self.D is None:
                self.D = len(gray_crop)
            else:
                assert self.D == len(gray_crop), "Dimensions size mismatch : {} / {}".format(self.D, len(gray_crop))

        self.N = len(self.dataGray)

        self.dataGray = numpy.vstack(self.dataGray).astype(numpy.float32)
        self.dataTheano = theano.shared(numpy.asarray(self.dataGray, dtype=theano.config.floatX), borrow=True)

        print(">>>", self.dataGray.shape)

        return True

    def _loadpkl(self):
        with open(os.path.join(self.path, self.name),'rb') as f:
            val = pickle.load(f)

        #epsilon = 1e-9
        self.dataGray = [self._patch(v) for v in val[0]]
        self.dataTheano = theano.shared(numpy.asarray(self.dataGray, dtype=theano.config.floatX), borrow=True)
        self.D, self.N = val[1], val[2]
        return True


    def _loadexr(self):
        import array
        import OpenEXR
        import Imath 

        os.chdir(self.path)
        data = []
        dataC = []

        for f in glob('*-img.exr'):
            f_mask = f.rsplit("-", 1)[0] + "-mask.exr"
            f_mask_hdl = OpenEXR.InputFile(f_mask)
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            Y = numpy.array(array.array('f', f_mask_hdl.channel("Y", FLOAT)).tolist())
             
            f_hdl = OpenEXR.InputFile(f)
             
            # Read the three color channels as 32-bit floats
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            (R,G,B) = [array.array('f', f_hdl.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
            R = numpy.array(R)
            G = numpy.array(G)
            B = numpy.array(B)

            dataC.append(numpy.dstack([R,G,B]))

            gray = numpy.ravel(self._rgb2gray(dataC[-1]))
            gray[Y<0.5] = 0
            gray_noNaNs = self._removeNaNs(gray)
            gray_crop = self._crop(gray_noNaNs)

            if self.D is None:
                self.D = len(gray_crop)
            else:
                assert self.D == len(gray_crop), "Dimensions size mismatch : {} / {}".format(self.D, len(gray_crop))

            self.dataGray.append(gray_crop)    
        
        self.dataGray = numpy.vstack(self.dataGray).astype(numpy.float32)

        self.N = len(self.dataGray)

        if not self.savepkl is None:
            with open(self.savepkl,'wb') as f:
                pickle.dump((self.dataGray, self.D, self.N), f, -1)

        
        self.dataTheano = theano.shared(numpy.asarray(self.dataGray, dtype=theano.config.floatX), borrow=True)
        return True


    def load(self):
        if self.type == 'MAT':
            return self._loadmat()
        elif self.type == 'EXR':
            return self._loadexr()
        elif self.type == 'PKL':
            return self._loadpkl()
        else:
            print("Format d'image '{}' inconnu!".format(self.type))
            return False




"""
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """

# start-snippet-1
class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


            

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """

        return self.theano_rng.normal(size=input.shape, avg=1., std=corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate, costInit=None):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        if costInit is None:
            tilde_x = self.get_corrupted_input(self.x, corruption_level)
            y = self.get_hidden_values(tilde_x)
            z = self.get_reconstructed_input(y)
            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            L = 0.5 * T.sum( T.abs_(z - self.x) , axis=1)
            #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            # note : L is now a vector, where each element is the
            #        cross-entropy cost of the reconstruction of the
            #        corresponding example of the minibatch. We need to
            #        compute the average of all these to get the cost of
            #        the minibatch
            cost = T.mean(L)
        else:
            cost = T.mean(costInit)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates, gparams)





def train_AE_multiplelayers(params):
    hidden_layers = params.hiddensize
    epoch_max = params.epochs
    dtype, dpath, dname = params.datasettype, params.datasetpath, params.datasetname
    learning_rate = params.lrate

    dI = DatasetImporter(dtype, dpath, dname, nanbehavior=params.nan, cropWidth=params.crop, savepkl=None if params.dumpparsedimgto == "" else params.dumpparsedimgto, patchHidden=True)


    #dI = DatasetImporter('MAT', "../data/20130823/", "envmap.exr.mat", nanbehavior='remove')
    #dI = DatasetImporter('EXR', "../data/", "", nanbehavior='zeros', cropWidth=20)
    #dI = DatasetImporter('PKL', "../data", "data2.pkl")
    dI.load()

    data, D, N = dI.dataTheano, dI.D, dI.N
    
    index = T.lscalar() 
    x = T.matrix('x')

    batch_size = 20
    n_train_batches = N // batch_size
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    input_vals = data

    liste_params = []
    liste_ae = []
    i = 0

    for hl in hidden_layers:
        ep = epoch_max
        print 'Begin layer'
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=D,
            n_hidden=hl
            )
        liste_ae.append(da)

        cost, updates, grads = da.get_cost_updates(
            corruption_level=0.3,
            learning_rate=learning_rate
            )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: input_vals[index * batch_size: (index + 1) * batch_size]
            }
        )

        start_time = time.time()
        # go through training epochs
        for epoch in xrange(ep):
            # go through trainng set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
            print 'Time from start : ', time.time()-start_time

        end_time = time.time()

        print 'Time for this layer : ', end_time-start_time

        liste_params.append(da.params)
        with open('params_op_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump( [(da.W.get_value()), da.b.get_value(), da.b_prime.get_value()], f, -1)

        new_inputs = da.get_hidden_values(input_vals).eval()
        #new_inputs_eval = new_inputs
        #print(new_inputs_eval.shape)

        # La sortie de cette couche devient l'entree de la prochaine
        input_vals = theano.shared(numpy.asarray(new_inputs,
                                        dtype=theano.config.floatX),
                                        borrow=True)    # Ca fait quoi ce parametre la?

        D = hl
        i += 1

    return

    # Fine tuning
    for ep in range(10):    # Nombre d'epochs pour le fine tuning

        for img in dI.dataGray:
            imgKeep = img
            input = img
            for layer in liste_ae:
                input = layer.get_hidden_values(input)
            for layer in reversed(liste_ae):
                input = layer.get_reconstructed_input(input)
            output = input.eval()

            diff = output - imgKeep
            for layer in reversed(liste_ae):
                cost, updates, grads = layer.get_cost_updates(
                    corruption_level=0.0,
                    learning_rate=learning_rate,
                    costInit=theano.shared(numpy.asarray(diff, dtype=theano.config.floatX), borrow=True)
                    )
                imgT = theano.shared(numpy.asarray(img, dtype=theano.config.floatX), borrow=True)
                train_da = theano.function(
                    [imgT],
                    cost,
                    updates=updates,
                    givens={
                        x: img
                    }
                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser Autoencoder')
    parser.add_argument('hiddensize', type=int, nargs='+', help="Taille des couches cachees")
    parser.add_argument("--datasetpath", type=str, help="Dossier contenant le dataset (path)")
    parser.add_argument("--datasetname", type=str, default="", help="Nom du fichier (ou pattern)")
    parser.add_argument("--datasettype", type=str, default="MAT", help="MAT | EXR | PKL")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'epoques pour chaque couche")
    parser.add_argument("--bruit", type=float, default=0.2, help="Bruit ajoute (std.var. de la gaussienne)")
    parser.add_argument("--lrate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--nan", type=str, default="zeros", help="zeros | remove")
    parser.add_argument("--crop", type=int, default=0, help="Nombre de pixels a retirer de chaque cote")
    parser.add_argument("--dumpparsedimgto", type=str, default="", help="Fichier dans lequel faire un dump des images traitees")
    args = parser.parse_args()

    train_AE_multiplelayers(args)


