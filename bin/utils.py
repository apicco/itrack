import os 
import numpy as np
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
	def myecc( t , d , j , p = 0.2 ) :
	
		u02 = t.mol()
		u20 = t.f_err()
		u11 = t.mol_err()

		# eccentricity, ellipse radii:
		l_1 = [ np.sqrt( ( u02[i] + u20[i] + np.sqrt( (u20[i] - u02[i]) ** 2  + 4 * u11[i] ** 2 ) ) / 2 ) for i in range( len( u02 ) ) ]
		l_2 = [ np.sqrt( ( u02[i] + u20[i] - np.sqrt( (u20[i] - u02[i]) ** 2 + 4 * u11[i] ** 2 ) ) / 2 ) for i in range( len( u02 ) ) ]
		
		N = len( l_1 )
		# ration between ellipse radii:
		l_r = [ l_2[ i ] / l_1[ i ] for i in range( N ) ]
		e = [ np.sqrt( 1 - l_r[ i ] ) for i in range( N ) ] #because of their above definitions l_2[ i ] < l_1[ i ] for each i
		
		NN = np.count_nonzero( ~ np.isnan( d[ 1 , : ] ) )

		#compute the distribution of min distances. The rational is that the best trajectories
		# are well isolated from their neighbours
		pp = np.zeros( NN )
		for i in range( NN ) :

			pp[ i ] = np.nanmin( d[ i , : ] )

		p = [ pp[ i ] / np.nansum( pp ) for i in range( NN ) ]

		#does removing the distance of the j-th trajectory from its nearest neighbor 
		#improves (i.e. decrease) the Shannon entropy? If yes, that's a good candidate
		#trajectory to reject. The Shannon entropy imporvement is measure as the decrease
		#of the ratio over the max Shannon entropy for an equal distance sample size (the 
		#max Shannon entropy is if all distances are equal.
		Sh = np.nansum( [ - ( p[ i ] ) * np.log( p[ i ] )  for i in range( NN ) if ( i != j ) ]  ) 
		Sh_max = np.nansum( [ - 1 / ( NN - 1 ) * np.log( 1 / ( NN - 1 ) ) ] * ( NN - 1 ) )
		rSh = Sh / Sh_max
#### TO Correct: it is still not working very well. The rSh changes a lot within frames, also for late frames it is nan or everything goes down to low values of rSh. I suspect there is a problem in the function dist_matrix

#### TO DO ->		# try to do the same thing but using the eccentricity. both the max and min eccentricity of each spot to compute two sets of probabilities and sh?

#		print( 'len( p ) : ' + str( len( p ) ) )
#		print( 'len( N * p ) : ' + str( len( N * p ) ) )
#		print( 'N : ' + str( N ) )
#		print( '< p > : ' + str( np.mean( p ) ) )
#		print( '< p**2 > : ' + str( np.std( p ) ) )
#		print( '< l_1/l_2 > : ' + str( np.mean( l_r ) ) )
#		print( 'med( l_1/l_2 ) : ' + str( np.median( l_r ) ) )
		print( '< (l_1/l_2)**2 > : ' + str( np.std( l_r ) ) )
		print( 'e : ' + str( np.mean( e ) ) )
		print( 'min( e ) : ' + str( min( e ) ) )
		print( 'max( e ) : ' + str( max( e ) ) )
#		print( 'min( l_1/l_2 ) : ' + str( min( l_r ) ) )
#		print( 'max( l_1/l_2 ) : ' + str( max( l_r ) ) )
#		print( 'E p : ' + str( sum( p ) ) )
#		print( 'max Sh : ' + str( max( Sh ) ) )
#		print( 'pred. max Sh : ' + str( N * max( Sh ) ) )
#		print( 'pred. min Sh : ' + str( N * min( Sh ) ) )
		print( 'Sh : ' + str( Sh ) )
		print( 'max Sh : ' + str( Sh_max ) )
		print( 'rSh : ' + str( rSh ) )
#		print('----------' )
		
		return( [ max( e ) , rSh ] )	
	#-------------------------------------
	def distance( x , y ) :

		return( np.sqrt( ( x[ 0 ] - y[ 0 ] ) ** 2 + ( x[ 1 ] - y[ 1 ] ) ** 2 ) )

	def centroid( x ) :

		return( [ np.nanmean( x.coord()[ 0 ] ) , np.nanmean( x.coord()[ 1 ] ) ] )

	def dist_matrix( tt , f ) : 
	
		l = len( tt )
	
		d = np.zeros( ( l , l ) ) * np.nan

		for i in range( l ) :

			if ( f >= tt[ i ].frames()[ 0 ] ) & ( f <= tt[ i ].frames()[ -1 ] ) :

				for j in range( i + 1 ) :
				
					if ( f >= tt[ j ].frames()[ 0 ] ) & ( f <= tt[ j ].frames()[ -1 ] ) :

						if ( i != j ) : 
	
							d[ i , j ] = distance( centroid( tt[ i ] ) , centroid( tt[ j ] ) )
							d[ j , i ] = d[ i , j ]

		return( d )
	#-------------------------------------
	def plot_traj( tt , f , what , ms = 3 , lw = 1 ) :

		cmap = cm.get_cmap( 'prism' , len( tt ) ) #color map
		
		shift = 0.5 #correct PT shift

		l = len( tt )

		d = dist_matrix( tt , f ) 

		for j in range( l ) :
		
			c = cmap( j / ( l - 1 ) ) #colormap
			t = tt[ j ]

			if ( f >= t.frames()[ 0 ] ) & ( f <= t.frames()[ -1 ] ) :

				#selected frames
				s = list( range( t.frames()[ 0 ] , f + 1 ) ) 
				sel = [ i for i in range( len( s ) ) if s[ i ] in t.frames() ]
		
				u = t.extract( sel )

				if ( what == 'coord' ) :
					
					if ( j == 3 ) :

						plt.plot( u.coord()[ 1 ] + shift , u.coord()[ 0 ] - shift , '-' , color = '#ffffff' , linewidth = lw + 1 )
						plt.plot( u.coord()[ 1 ] + shift , u.coord()[ 0 ] - shift , '-' , color = '#000000' , linewidth = lw + 1 )

					else :
						plt.plot( u.coord()[ 1 ] + shift , u.coord()[ 0 ] - shift , 'o' , color = c , markersize = ms )
						plt.plot( u.coord()[ 1 ] + shift, u.coord()[ 0 ] - shift , '-' , color = c , linewidth = lw )

				#elif ( 1 < j <= 4 ) :
				else :
				
					if ( j == 3 ) :

						plt.plot( myecc( t , d , j )[ 0 ] , myecc( t , d , j )[ 1 ] , 'o' , color = '#ffffff' , markersize= ms + 1 )
						plt.plot( myecc( t , d , j )[ 0 ] , myecc( t , d , j )[ 1 ] , 'o' , color = '#000000' , markersize = ms )

					else :

						plt.plot( myecc( t , d , j )[ 0 ] , myecc( t , d , j )[ 1 ] , 'o' , color = c , markersize = ms )

					plt.xlim( 0 , 1 )
					plt.ylim( 0 , 1 )
					plt.xlabel( 'e_M' )
					plt.ylabel( 'Sh / max( Sh )' )
	
		return u.f_err()
	#-------------------------------------

	#load the movie
	im = tiff.imread( path_movie )

	gs = GridSpec( 4 , 2 )

	for f in range( 0 , len( im ) ) :

		frameName = path_frame + 'frame' + '%04d' % f

		plt.figure( figsize = ( 8 , 7 ) )
		movie = plt.subplot( gs[ 0 : 2 , 0 : 2 ] )
		plt.imshow( im[ f , : , : ] , cmap = 'gray' )
	

		plot_traj( tt , f , 'coord' ) 

		plt.xlabel( 'Pixel' )
		plt.ylabel( 'Pixel' )
		movie.set_aspect( 'equal' )


		ecc = plt.subplot( gs[ 2:4 , 0:2 ] )
		plot_traj( tt , f , 'ecc' ) 

		plt.savefig( frameName )
		
		plt.close()
		

