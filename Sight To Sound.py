from math import sqrt, pi, log2, sin
import numpy as np
from matplotlib.pyplot import plot, show, axis, grid
from PIL import Image,ImageOps
import wave

# We can think of each point as a point in the co-ordinate plane.
# for n = 1, the figure will have 4^1 points. i.e. 4 points
# for n = 2, the figure will have 4^2 = 16 points
# for n = 3, the figure will have 4^3 = 64 points
# ...
# for n = n, the figure will have 4^n points

# We can divide the answer figure into 4 portions (quadrants). For example
# If n = 3, we will have 4^3 = 64 points => sqrt(64) = 8 rows, sqrt(64) = 8 columns
# Each quadrant will have sqrt(64)//2 = 4 rows and 4 columns
# . . . .| . . . .
# . . . .| . . . . 
# quad I | quad II
# . . . .| . . . .
# . . . .| . . . .
# _ _ _ _ _ _ _ _ 
# . . . .| . . . .
# . . . .| . . . .
#quad III| quad IV
# . . . .| . . . .
# . . . .| . . . .

# The function phc (pseudo hilbert curve) takes value of some positive integer n as input where n is related to number of pixels in the image as:
# n = 1 ⇒  21∗21=4  pixels
# n = 2 ⇒  22∗22=16  pixels
# n = 3 ⇒  23∗23=64  pixels
# n = 4 ⇒  24∗24=256  pixels
# Thus for an image of x * x pixels we need a pseudo hilbert curve with n =  log2(x) .

def phc(n):
	"""
	Input: n 
	Output: list of coordinates on edges on the pseudo hilbert curve for images with 2**2n pixels. 
	"""
	# Trivial case
	if n == 1:
		return [(1,1),(2,1),(2,2),(1,2)]
	else:
		n_rows = int(sqrt(4**n)) #number of rows = sqrt of total number of points 
		L = phc(n-1)

		# We can divide the answer figure into 4 portions(quadrants).
		# Quadrant 1 is the image of f(n-1) rotated by -90 degrees i.e. elements of f(n-1) with points interchanged (if (1,2) is in f(n-1), => (2,1) will be in quad1) 
		quad1 = [] 
		for x in range(len(L)):
			quad1.append(tuple([L[x][1], L[x][0]])) 

		# Quadrant 2 is the mirror image of quad1 about the vertical, => row number of each point corresponding to a pt in quad1 will remain the same.
		quad2 = [0 for x in range(len(quad1))]
		for x in range(1, len(quad2)+1):
			quad2[-1*x] = tuple([quad1[x-1][0], n_rows + 1 - quad1[x-1][1] ])

		# Quadrant 3 is the image of f(n-1) shifted towards the bottom => row number of each point corresponding to a pt in quad1 will be incremented. 
		quad3 = []
		for x in range(0, len(L)):
			quad3.append(tuple([L[x][0] + n_rows//2, L[x][1]]))


		# Quadrant 4 is the image of f(n-1) shited towards the bottom and then to the right by as many units as there are rows in each quadrant. 
		quad4 = []
		for x in range(len(L)):
			quad4.append(tuple([L[x][0] + n_rows//2, L[x][1] + n_rows//2]))

		# The final image of f(n) is drawn in the order: quadrant 1 -> quadrant 3 -> quadrant 4 -> quadrant 2
		# So the final order of elements for the image of f(n) becomes:
		answer = quad1 + quad3 + quad4 + quad2

	return answer

def find_n(x):
  """
  input: x,y where x,y is size of image in pixels.
  x must be equal to y, and log2(x) must be a natural number.  
  output: corresponding value of hilbert curve
  """
  lg = log2(x)
  if lg == int(lg):
    return int(lg)
  return "blah"

I = Image.open("heart.jpg", mode="r")
blackI = ImageOps.grayscale(I)
pixels = np.asarray(blackI)
x,y = blackI.size


def map_phc(mat):
	"""maps each pixel in the image to a corner in it's respective hilbert curve"""
  L = []
  # x,y = mat.size
  n = find_n(x)
  if n:
    for coord in phc(n):
      L.append(mat[coord[0]-1][coord[1]-1])
  return L
output = map_phc(pixels)

# We can create sine waves to make individual frequencies. The parameters for which will be:
# 1. F ( frequency ) 
# 2. A ( amplitude )
# 3. T ( time )

# We will use sample rate = 44100 hz. 44.1 kHz, or 44,100 samples persecond, is the most 
# popular sample rate used in digital audio, especially for music content. 44.1 kHz allows 
# reproduction of all frequency content below 22.05 kHz which covers all frequencies heard by a normal person.


def generateSine(F, A, T):
	return (sin(T/44100*2*pi*F)*A)

num_seconds = 1 # length of audio 
lowest_frequency = 220 #  in hz
frequency_n = 72 #not in hez
#  A value that directly detirmines the frequency
#  Find the highest frequency with this:
#  2**(255/frequency_n)*lowest_frequency
#  That puts the max frequency at just under 2562hz
# frequency_n = 72 (Not in hz)

# Generate the Audio output list
outputAudio = [0]*int((44100*num_seconds)) #zero array
len(outputAudio)


pixel_count = 0
for pixel in output:
	# if pixel_count % 10 == 0:
	print (pixel_count,"pixels completed out of", x*y, "          ", end="\r")
	pixel_count += 1
	frequency = 2**(pixel/frequency_n)*lowest_frequency;
	for t in range(0,len(outputAudio)):
		# Constantly layer the audio on top of one another until it's complete
		# We use a really big offset to t becuase if we don't we get a really weird tone that quickly
		# Tapers off. 
		outputAudio[t] += generateSine(frequency,1/(x*y/2),t+524288)

for i in range(0,len(outputAudio)):
	# Add one
	outputAudio[i] += 1
	outputAudio[i] = outputAudio[i] * 16384

	# Convert it to an integer
	outputAudio[i] = int(outputAudio[i])


print("Writing file to disk, also converting the sound data to bytes.")
# Open the output file
audio_output = wave.open("output1.wav", mode = 'wb')
print("a")
# Set it to one channed 16 bit 44100hz no compression
audio_output.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
print("b")
# Write the output to the file
for i in outputAudio:
	audio_output.writeframes(abs(i).to_bytes(2,'little'))