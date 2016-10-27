<!DOCTYPE html>
<html>
<head>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [["$","$"],["\\(","\\)"]]}
  });
   MathJax.Hub.Config({ TeX: { equationNumbers: {autoNumber: "AMS"} } });
</script>
<title>2D waves using circles</title>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

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

		<h1 class='special'>2D waves using circles</h1>
		A while ago I saw this nice animation (not sure what is the original source, but it can be found in different places on the web) demonstrating a propagation 
		of what seems like a compression wave composed only from the rotation of black dots on closely placed circles, creating the illusion of a wave form with amplitude perpendicular
		to the screen (transverse wave).

		<center><img src='images/circwave.gif' width='30%' height='30%'/></center>
		<br>

		Recently I saw it again, and decided to check if I can reproduce it and try to better understand how the effect is generated.
		So after examining more carefully the image, you see that there are equally spaced circles, intersecting each other, with a black dot
		on the perimeter which is rotating with a constant angular velocity in all the circles. Further, one can see that in each consecutive
		circle, the black dot is slightly shifted in its path around the circle. So how does this result in a propagating wave effect?
		A longitudinal plane wave in 2D can be written in the complex representation:
		\begin{equation}
			A(x,y,t)=A_0e^{i(\omega t -\vec{k}\cdot\vec{r})}
		\end{equation}

		where $A_0$ is a complex amplitude (real amplitude and phase), $\vec{k}$ is the wave vector, and $\vec{r}$ is position in 2D.
		This describes a plane wave propagating in the direction $\vec{k}$.
		At a given position the wave given by $e^{i(\omega t+\phi)}$ is nothing more than a rotation in the complex plane, composed
		of harmonic motion $cos(\omega t+\phi) , sin(\omega t +\phi)$ along the $x$ and $y$ axes. The position dependence given by 
		$\vec{k}\cdot\vec{r}=k_x x+k_y y$, describes the phase shift between adjacent points in the x and y directions, and the variation
		is given by the wave vector components.<br>
		With this in mind, I managed to generate such an animation in Mathematica using the code below:
		<pre>
		<code>
n = 20;  (* number of circles per row and column*)
w = 2 Pi/60;  (* angular frequency *)
kx = 2 Pi/15; ky = -2 Pi/15; (* wave vector *)

v = Table[
(* draw a grid of circles of radius 0.115 with separation 0.115 along x and y*)
Panel[Graphics[{{Table[Table[Circle[{(i - 1) 0.115, 0.115 j}, 0.1], {i, 1, n}], {j, 1, n}]},   
(* draw black dots on the circles include in the x,y coordinates the time and phase.*)
{Table[Table[Disk[{(i - 1)*0.115 + 0.1*Cos[w t + kx + ky j], 0.115 j + <br>
	0.1*Sin[w t + kx i + ky j]}, 0.017], {i, 1, n}], {j, 1, n}]}},  
(* range of the graphics in order for the frame to be fixed, and won't move *)
PlotRange -> {{-0.15, 2.3}, {0, 2.45}}, Background -> White] ], 
{t, 0, 2 Pi/w, 2*Pi/w/64}]; (* the sequence of images set by t, in one period, with 64 frames. *)

Export["cw1.gif", v, "DisplayDurations" -> 0.03,ImageSize -> 400];  (* export to gif *)
		</code>
		</pre>

The resulting animation, for the chosen parameters, and especially the chosen wave vector, which points in the direction from top left to bottom right.

		<br>
		<center><a href='images/cw11.gif'><img src='images/cw11.gif'/></a></center>
		<br>

By choosing different values for $k$ we can control both the wavelength (distance between the dark or compressed patches) 
as well as the direction of propagation.

Below I show a few examples:

<center>
<table>
	<tr>
		<td style='text-align:center'>$k_x=2\pi/20, k_y=0$</td>
		<td style='text-align:center'>$k_x=2\pi/20, k_y=2\pi/20$</td>
		<td style='text-align:center'>$k_x=k_y=2\pi/50$</td>
	</tr>
<tr>
<td> 
		<center><a href='images/cw2.gif'><img src='images/cw2.gif' /></a></center>
</td>
<td> 
		<center><a href='images/cw3.gif'><img src='images/cw3.gif' /></a></center>
</td>
<td> 
		<center><a href='images/cw4.gif'><img src='images/cw4.gif' /></a></center>
</td>
</tr>
</table>
</center>

We see that the wave vector corresponds indeed to the direction of propagation and wavelength ($\lambda=2\pi/k$), and the 
wavelength is large enough the motion of the black dots looks more in phase as the wavelength is greater than the size of
the sample shown.


Finally, it is also nice to see this wave pattern by showing only the black dots without the circles, then it's harder to see
that the origin for this motion is in the constant circular motion of the black dots.


		<center><a href='images/cw5.gif'><img src='images/cw5.gif' width='40%' height='40%'/></a></center>

</div>


</body>


</html>