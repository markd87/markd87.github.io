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
<title>Julia sets</title>
<!--<link rel="stylesheet" href="style.css" type="text/css">-->
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

		<h1 class='special'>Julia sets</h1>


Consider the following "game" where given a complex valued function $f(z)$ and some point in the complex plane $z_0$
we generate a set of numbers $z=\{z_0,f(z(0)),f(f(z_0)),..\}$ which is given by the rule $z_{n}=f(z_{n-1})$. This describes a 
discrete dynamical system over the complex field.
A possible question to ask regarding this process is whether it is bounded $(\exists M$ such that $f^n(z_0) \lt M, \forall n)$ or not.
The set of all points $z$ for which the dynamical process described is bounded, is called the filled Julia set of f. The boundary
of the filled Julia set is called the Julia set of f. Named after the French mathematician <a href="http://en.wikipedia.org/wiki/Gaston_Julia"  target='_blank'>Gaston Julia</a>.<br>
The reason I decided to write about these sets is that they typically draw beautiful and complex fractals in the complex plane. <br>
The procedure for plotting the resulting Julia set for a given $c$ is to consider a region in the complex plane with a given number of pixels
(width x height) where each pixel is a number in the complex plane. Then by going over each pixel in the region, and applying the function
$N$ times, we check each time if the resulting complex number has an absolute value greater than some escape radius $r_{esc}$
(2 is enough), in this case we can be certain that this initial $z_0$ will escape to infinity under $f$ and that $z_0$ is not in the 
Julia set. If we reached N without having $|z|\geq r_{esc}$ then it is possible that $z_0$ is in the set.
We can't be certain, but if we choose $N$ big enough, then the plot will be more accurate. Also by using a higher resolution (more pixels for a given region)
Then more details will be visible in the plot. <br>
One type of Julia sets is the one originating from the function $f(z)=z^2+c$ where $c$ is a constant complex number.
Below I show a number of examples for different choices of $c$. The filled Julia set is in black, and the colors indicate the number
of time or iterations of $f$ before a given $z_0$ escaped to infinity. This gives a nice visualization of the dynamical system.
The Julia set itself is in the boundary between the black regions and the colored regions, where the inner region contains points which are bounded
and are repelled from the boundary inwards, whereas the outer region is repelled from the boundary towards infinity. <br>
An additional trick to make the images nicer, is to make the color scheme smooth. Since as described the coloring is based on the 
number of iterations required to reach the escape radius, this number is an integer, and we will get a step like coloring which is not
very appealing. It is possible to have a mapping between the integer number of iterations to a real value which changes smoothly
and allows the coloring to be continuous and smooth. <br>
The images below show the region $x\in [-2,2],y\in [-1.33,1.33]$ for 8 different values of $c$. The filled Julia set is in black, and the colors
going from red to blue indicate a decreasing number of iteration to reach to the escape radius which was chosen here to be $30$ as it also
allows for a smoother transition of colors. In the bottom row, where there is no evident black filled region, it is because the points
within the Julia set are along thin lines buried inside the other colored features (The Julia set is not connected in these cases), and therefore are not seen, and we only see
a dark red color which indicates a large number of iterations required to reach infinity, and this is close to the Julia set.

<br>
<br>

<table class='aligncenter' id='drums'>
	<tr>
		<td>$c=-1$</td>
		<td>$c=-0.81+0.1795i$</td>
		<td>$c=-0.62772+0.42193i$</td>
		<td>$c=0.300283+0.48857i$</td>
	</tr>
	<tr>
		<td><a href='images/j1.png'><img src="images/j1.png" width='80%'/></a></td>
		<td><a href='images/j2.png'><img src="images/j2.png" width='80%'/></a></td>
		<td><a href='images/j3.png'><img src="images/j3.png" width='80%'/></a></td>
		<td><a href='images/j7.png'><img src="images/j7.png" width='80%'/></a></td>
	</tr>
	<tr>
		<td>$c=-0.835-0.2321i$</td>
		<td>$c=-0.513+0.5212i$</td>	
		<td>$c=0.285+0.013i$</td>
		<td>$c=0.285+0.013i$</td>
	</tr>
	<tr>
		<td><a href='images/j4.png'><img src="images/j4.png"  width='80%'/></a></td>
		<td><a href='images/j5.png'><img src="images/j5.png"  width='80%'/></a></td>
		<td><a href='images/j6.png'><img src="images/j6.png"  width='80%'/></a></td>
		<td><a href='images/j8.png'><img src="images/j8.png"  width='80%'/></a></td>
	</tr>
</table>

<br>
<br>


Examining the various Julia sets, we can notice that in some cases the filled Julia set is connected and in some disconnected.
One can ask for what values of $c$ is the Julia set connected. The set of numbers $c$ for which the Julia set is connected
is called the Mandelbrot set, named after the mathematician <a href="http://en.wikipedia.org/wiki/Benoit_Mandelbrot" target='_blank'>Benoit Mandelbrot</a>. 
There is a theorem which states that for a given $c$, the corresponding Julia set of $f$ is connected
if $z=0$ is in the filled Julia set of $f(z)=z^2+c$, and disconnected otherwise. This allows to obtain the Mandelbrot set
by varying $c$ over the complex plane and starting with $z=0$, check whether the process diverges or not, if not than $c$ 
is in the Mandelbrot set as it was defined.

<br>
<br>
<center><a href='images/mandelbrot.png'><img src="images/mandelbrot.png"/></a></center>
<br>
<br>



The images were generated using Python: <a href='julia.py' id='code'>code</a>.
</div>


<script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>

</body>


</html>