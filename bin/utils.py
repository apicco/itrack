import os 

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm 
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use( [ 'seaborn-pastel' , 'movie_noGrid' ] )

from trajalign.traj import Traj
from skimage.external import tifffile as tiff

def load_traj( path , pattern = 'x' , comment_char = '%' , **kwargs ) :
	"""
	load all the trajectories identified by pattern in path
	"""

	output = []

	#list the trajectory files identified by pattern
	r = [ f for f in os.listdir( path ) if pattern in f ]

	#load the trajectory as a Traj object
	for i in range( 0 , len( r ) ) :
		
		t = Traj()
		t.load( path + '/' + r[ i ] , comment_char = '%' , **kwargs )#, *attrs )
		#t.extract( t.f() == t.f() ) #remove NA
		output.append( t )

	return output

def show_on_movie( tt , path_movie , path_frame ) :
	
	#-------------------------------------
	def plot_traj( tt , f , ms = 3 , lw = 1 ) :

		cmap = cm.get_cmap( 'prism' , len( tt ) ) #color map
		
		shift = 0.5 #correct PT shift
		
		l = len( tt )
		for j in range( l ) :
		
			c = cmap( j / ( l - 1 ) )
			t = tt[ j ]

			if ( f >= t.frames()[ 0 ] ) & ( f <= t.frames()[ -1 ] ) :

				#selected frames
				s = list( range( t.frames()[ 0 ] , f + 1 ) ) 
				sel = [ i for i in range( len( s ) ) if s[ i ] in t.frames() ]
			
				u = t.extract( sel )
				plt.plot( u.coord()[ 1 ] + shift , u.coord()[ 0 ] - shift , 'o' , color = c , markersize = ms )
				plt.plot( u.coord()[ 1 ] + shift, u.coord()[ 0 ] - shift , '-' , color = c , linewidth = lw )
	#-------------------------------------

	#load the movie
	im = tiff.imread( path_movie )

	for f in range( 0 , len( im ) ) :

		frameName = path_frame + 'frame' + '%04d' % f
	
		plt.figure( 1 , figsize = ( 7 , 7 ) ) 
		plt.imshow( im[ f , : , : ] , cmap = 'gray' )
		
		plot_traj( tt , f ) 

		plt.xlabel( 'Pixel' )
		plt.ylabel( 'Pixel' )

		plt.savefig( frameName ) 
		plt.close()
		

