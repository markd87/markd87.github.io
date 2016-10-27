<!DOCTYPE html>
<html>
<head>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [["$","$"],["\\(","\\)"]]}
  });
   MathJax.Hub.Config({ TeX: { equationNumbers: {autoNumber: "AMS"} } });
</script>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<title>Fourier Transform</title>

    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ==" crossorigin="anonymous">
    
<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-70400930-1', 'auto');
  ga('send', 'pageview');

</script>
    <style>

      p{
        text-align: justify;
      }

      .navbar{
        border-bottom: 1px solid #d3d3d3;
      }
      .navbar{
        margin-bottom:0px;
        padding:5px;
      }

      .navbar ul{
        float:right;
        margin-top: 10px;
      }

      .navbar-brand{
        font-size:2em;
        color:#555 !important;
        line-height:37px;
      }

      .active{
        border:1px solid grey;
        background-color: none;
      }

      #banner{
        background-image: url(img/backs.jpg);
        background-position: 75% 68%;
        background-size: cover;
        background-repeat: no-repeat;
        width:100%;
        color:#eee;
        border-bottom: 1px solid #d3d3d3;
        margin-top:0px;
      }

      .bannerText{
        font-size:1.5em;
        width:60%;
        font-weight: bold;
      }

      .mail{
        margin-top:-2px;
      }
      .ic img{
        width:18px;
        margin-top:-2px;
      }

      .linkedin img{
          width:25px;
          margin-top:-2px;
      }

      #blender_img{
        text-align: center;
      }

      .footerp{
        color:#222;
        margin-bottom:5px;
        margin-top:10px; 
        text-align: center;
      }

      #footer{
        padding-top:40px;
        background-color: #f8f8f8;
        width:100%;
        text-align: center;
        border-bottom: 1px solid #d3d3d3;
      }

      .contact li{
        padding-right:10px;
      }


      .page{
        padding-top:70px;
      }

      #projects{
        margin-bottom:30px;
      }

      .break{
        clear:both;
        width:100%;
        border-bottom: 1px solid #d3d3d3;
      }
    </style>

</head>

  <body data-target=".navbar" data-offset="50">

    <nav class="navbar navbar-default navbar-fixed-top">
      <div class="container" id="navigation">
        <div class="navbar-header">
          <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
            <span class="sr-only">Toggle navigation</span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
          </button>
          <img src="https://i1.rgstatic.net/ii/profile.image/AS%3A287718316756997%401445608800899_l/Mark_Danovich.png" style="
    height: 70px;
    border-radius: 35px;
    float: left;
    margin-right: 20px;
          ">
<!--<a class="navbar-brand" href="">MDD</a>-->
        </div>
        <div id="navbar" class="navbar-collapse collapse">
          <ul class="nav navbar-nav right">
            <li><a href="../index.html">home</a></li>
            <li ><a href="../about.html">about me</a></li>
            <li><a href="../research.html">research</a></li>
            <li><a class="active" href="../stuff.html">random</a></li>
          </ul>
        </div>
      </div>
    </nav> 

<div id='content' class='container lead' style='text-align:justify; margin-top:100px;'>
		<h1 class='special'>Fourier Transform</h1>

		Fourier analysis is a central tool in signal processing. It provides a representation in the frequency domain of the signal which is usually given in the time domain, thus decomposing the time signal into a sum of oscillatory components of single frequency which describe the variation in the original signal.
		Here I will show a simple example of using the Fourier Transform for a real signal - a sound signal, whose pressure waves amplitudes were transformed into voltage values which then are represented digitally in a file on the computer as a discrete sequence of values, each taken at a constant interval called 
		the sampling period for which there is a corresponding sampling frequency, and these values are (up to quantization error) match the original sound at a given time. 
		
		The Fourier Transform usually introduced for continuous valued functions on $(-\infty,\infty)$ provided that 
		$\int_{-\infty}^{\infty} |f(x)|dx<\infty$, $f(x)$ is bounded and has a finite number of discontinuities in the form:

		\begin{equation}
			F(\omega)=\int_{-\infty}^{\infty}f(t)e^{-2\pi i \omega t}dt \\
			f(x)=\int_{-\infty}^{\infty}F(\omega)e^{2\pi i \omega t}d\omega
		\end{equation}

		When considering signals represented in a computer, we are dealing with discrete signals and normally finite in time.
		In this case we consider a signal of length $N$. As the Fourier Transform is essentially a change of basis to a basis of 
		oscillatory function we first construct a basis of periodic functions on $[0,N-1]$ for the vector space $\mathbb{C}^N$,
		$w_k[n]=e^{i\omega_k n}$, $n=0,..,N-1$. Requiring whole number of periods within the domain we have $w_k[N]=w_k[0]=1$, which translates to
		$(e^{i \omega_k})^N=1$, which has N solutions corresponding to the N roots of 1 on the unit circle $e^{i 2\pi m/N}$, $m=0,..,N-1$. This can be 
		shown to be an orthogonal set of vectors and thus consisting a base in $\mathbb{C}^N$. The Fourier Transform in this case takes the following form (which is essentialy a change of basis):

		\begin{equation}		
			X[k]=\sum_{n=0}^{N-1} x[n] e^{-i\frac{2\pi n k}{N}}, \quad k=0,..,N-1 \\
			x[n]=\frac{1}{N}\sum_{k=0}^{N-1} X[k] e^{i\frac{2\pi n k}{N}}, \quad n=0,..,N-1
		\end{equation}

		Where the factor $1/N$ comes from normalization of the basis function to be orthonormal.
		Since we are dealing with real signals, and complex basis functions, we have two frequency components for each frequency,
		positive and negative, these add up with opposite phases such that the complex part of the basis function disappears and we get
		eventually a real signal back from the Fourier decomposition. The negative frequencies correspond to the frequencies above $\frac{2\pi}{N}\frac{N}{2}=\pi$
		Therefore when we study the spectrum of a real signal we can look only at the first half of the frequencies, since it is symmetric
		about the center.
		<br>

		I will first present the basic code with explanations for generating a Fourier transform of a signal in Matlab.
		<br>
		<br>
		<pre style='max-width:100%; word-wrap: break-word;'><code >
file='/Users/markd/Desktop/sounds/piano_a_note.wav';
[x,fs]=audioread(file);  %read sound file to x, and sampling frequency to fs
Ns = length(x); % get length of the signal sequence
t = (1/fs)*(1:Ns);  % get the duration of the signal knowing the sampling period 1/fs and the number of <br>%sample points

Xk = abs(fft(x));  % do the Fourier transform and take only the amplitude using abs()
Xk = Xk(1:Ns/2);   % take only the first half of the transform due to the symmetry about the center
f = fs*(0:Ns/2-1)/Ns;  % generate the vector of frequencies knowing the sampling frequency <br> %and reaching fs/2

% plot the waveform of the signal
figure('color','white')
plot(t, x,'color','blue')
xlabel('Time (s)')
ylabel('Amplitude')

%plot the Fourier transform of the signal
figure('color','white')
plot(f, Xk/max(Xk))
xlim([0 4000])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
		</code></pre>
		<br>

		I will now look at a few examples of performing a Fourier Transform on a real signal using Matlab. The examples will be with sound signals.
		The first example will be the sound wave produced from a tuning fork producing a close to pure tone at $440Hz$ or the note A. 
		Below are the sound file, the waveform and the Fourier transform of the signal.
		<br>
		<br>
		<center>
		<audio controls>
		 <source src="images440.mp3" type="audio/mp3">
		Your browser does not support the audio element.
		</audio>
		</center>
		<br>

		<img src='images/440wave.png' width='49.5%'/> <img src='images/440fourier.png' width='49.5%'/> 

		We see that there is a single dominant component at $442.4Hz$. <br>

		Next I will look at the signal produced by playing the middle A key ($440Hz$) on a piano.

		<br>
		<br>
		<center>
		<audio controls>
		 <source src="images/piano_a_note.wav" type="audio/wav">
		Your browser does not support the audio element.
		</audio>
		</center>
		<br>

		<center><img src='images/piano_a_note_wave.png' width='49.5%'/> <img src='images/piano_a_note_fourier.png' width='49.5%'/> </center>

		In this case, again we see a dominant frequency component at $439.1Hz$ as expected, but also several other components
		which didn't appear in the tuning fork example, despite playing the same note. However we see that the additional components
		appearing in the piano Fourier analysis are multiples of the fundamental frequency with different amplitudes. These are the fundamental
		frequencies of the vibrating string in the piano corresponding to $\sim 440Hz$. The presence of the additional harmonics is what
		gives the piano its characteristic sound and distinguishes it from the tuning fork, a guitar or a violin.


		Finally a more curious example, in which I will analyze the frequencies present in the famous opening chord of the song 
		"A Hard Day's Night" by the Beatles. This analysis was already thoroughly done in <a href='http://www.mscs.dal.ca/~brown/n-oct04-harddayjib.pdf'>here</a>, but 
		as an exercise I will try to reproduce these results.

		<br><br>
		<center>
		<iframe style='max-width:100%' src="//www.youtube.com/embed/zSm0M-BbVdY" frameborder="0" allowfullscreen></iframe>
		</center>
		<br>

		And the opening chord alone:

		<br>
		<center>
		<audio controls>
		 <source src="images/chord.ogg" type="audio/ogg">
		Your browser does not support the audio element.
		</audio>
		</center>
		<br>

		The Fourier transform:

		<center><img src='images/beatles.png' width='80%'/></center>

		As we see, there are many frequency components, some of which as we saw, are higher harmonics of lower frequency notes.
		In the chromatic scale there are 12 notes equally separated by semitones such that a difference of one octave (12 semitones) is equivalent
		to a doubling of the frequency. Therefore a semitone is a difference in frequency of $2^{1/12}$ and the number of semitones between a note $f$ and a reference note
		$f_0$ is given by $12\log_2{f_1/f_0}$. This is convenient for naming the notes relative to a reference note such as middle A on a piano $A4=440Hz$.
		The frequencies present in the analysis are: 
		74.82, 87.22, 110.4, 150, 175.2, 195.3, 218.5, 262.5, 299.7, 351.3, 392.5, 438.1, 524.9, 589, 686.2, 785.4, 886.6, 961.1, 981.9, 
		1051, 1084, 1186, 1286, 1321, 1386, 1489, 1578, 1632, 1751, 1969, 2203, 2368, 2639, 2764, 3084, 3148.
		These correspond to the following number of semitones difference from A4 (rounded): -31,-28, -24, -19, -16, -14, -12, -9,
		 -7, -4, -2, 0, 3, 5, 8, 10, 12, 14, 14, 15,
		  16, 17, 19, 20, 21, 22, 23, 24, 26, 28, 29, 31, 32, 34.
		And the corresponding notes: D2, F2, A2, D3, F3, G3, A3, C4, D4, F4, G4, A4, C5, D5, F5, G5, A5, B5, C6, D6, E6, ... <br>
		As described in the paper it is known that the instruments involved in the recording include a 12-string guitar, a 6-string guitar, a bass guitar and a piano.
		As can be seen in the analysis, several frequencies appear as doublets or triplets. This is explained in the linked paper as notes coming from a piano, in which
		the notes are produced by a hammer hitting one for the low notes, two or three (starting from C3 or D3) strings which should be tuned to the same note, but differences between the tuning can occur.
		Additional important information is regarding the 12-string guitar in which every string is doubled and both strings are one octave apart (A2,A3 for example).
		Using this knowledge the author in the paper is able to deduce which notes were played by which instrument. I note that the first two notes D2, F2 are not present in the analysis in the paper
		perhaps due to the filtering of small amplitudes components, These most likely come from either the piano or the bass guitar.
		I will conclude with the conclusion of the paper: the pairs A2,A3, E3,E3, G3,G4, C4,C5 were played on the 12-string guitar (Harisson), The triplets D3, F3, D5, G5, E6 were played on the piano, 
		one D3 was played on the bass guitar (McCartney), and a strong (high amplitude) C5 on the 6-string guitar (Lennon), and I will add that two lower notes D2, F2 were played either by the piano or the bass.
		All higher notes are most likely higher harmonics of the other notes played.
</div>


</body>


</html>