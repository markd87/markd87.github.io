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
		<h1 class='special'>The power of estimation</h1>
	
		One of the fastest ways to solve any problem is to know the answer in advance. 
		Especially in Physics, one of the most important tools to have is the abillity to estimate the answers to problems
		before actually solving them. This involves invoking physical or general considerations which allow to get
		an order of magnitude estimate to a given problem which can be physics related but can also be completely general. Sometimes this practice is also
		called back of the envelope calculation.
		Some of the greatest Physicists were famous in their ability to make such back of the envelope estimations of various problems,
		most famous of which were Fermi, Feynman and Landau.
		During my studies I took a course on methods of estimations and approximations taught by Prof. Nir Shaviv. In the first lecture
		he said that Physicists are like superman, they can solve any problem.
		Here I wanted to present the famous Fermi problem which shows how one can tackle a seemingly impossible problem using the power of estimations, and then I give a short 
		exmaple of the methods of estimation used in the context of a Physics problem.
		<br><br>

		The question Fermi used to present as the first problem was: "How many piano tuners are there in Chicago?"
		The approach to this and other seemingly impossible problems is through separating it to smaller problems which we
		can estimate more easilly the answer to and then construct the final answer, also known as 'divide and conquer'.
		The sub-problems that need to be solved are: <br>
		the number of people in Chicago: This is definatley more than 1 million and less than 100 million, we'll take 10 million.<br>
		frequency of piano tunings: definitley more than once a month and more than once per decate, we'll take once a year. <br>
		It takes about 2 hours to tune a piano (88 keys)<br>
		Number of pianos: Here there are a number of ways to estimate, with various resoultions. We can assume 3 people per household on average
		with 5% of households having a piano giving: $160,000$ pianos. <br>
		An alternative way is to consider how many people hold a piano, there are about 10% (less than 100% more than 1%) who play the piano
		but probably much less hold one, about 10% of them will also hold a piano, so we'll take $1\times 10^{-2}$. In addition there are pianos usually in churches and schools, these are
		about 1 per 1000 people, so additional $2\times 10^{-3}$, giving a total of 120,000 pianos. We can take the average of the two
		as the final estimate: 140,000 pianos.<br>
		The final estimation is the amount of time a piano tuner works: we take a standard of 5 days a week, 8 hours a day, 50 weeks a year,
		which is 2000 piano tuning hours a year. At 2 hours per piano this gives 1000 piano tunings per year.
		Since there are $140,000$ pianos to be tuned per year, there should be about 140 piano tuners.
		With a tool such as <a href='http://www.wolframalpha.com/input/?i=how+many+piano+tuners+are+there+in+Chicago'>wolfram alpha</a>
		This question can actually be answered exactly and quickly, giving ~290, which is of the same order of magnitude as our result. One
		can play with the estimates to see where the estimations can be improved.
		<br><br>
		A more sophisticaed example of Fermi's brilliant imployment of the power of estimation is the estimation of the energy released in the
		Trinity atomic bomb test in 1945 as part of the Manhattan project leading to the bombing of Hiroshima and Nagasaki less than a month later.
		The way he estimated it required a little more than just nubmer guessing but some actual data which he obtained by witnessing the
		explosion and the resulting shock wave itself. By throwing pieces of paper where he stood, he measured the displacment outwards and inwards
		of these pieces papers due to the passing shock front. He found that they were displaced by 2.5m, and he stood 16km from the place of explosion.
		The way to estimate the total energy is using conservation of energy and the work done by the shock wave on the air at the place where Fermi stood.
		The volume diplaced is the hemisphirical shell of thickness 2.5 m: $\Delta V=(2\pi r^2)d=4\times10^9 m^3$.
		The work done is $E=W=P\Delta V$, the pressure here is the over pressure above atmospheric pressure. 1 atmosphere is
		$10^5N/m^2$, which is like 10 tons of weight on a human body, this is not reasonable, since in that case Fermi wouldn't be able to 
		withstand it. It is probably more than $10^2 N/m^2$ or 10kg, so we can take $P=10^3 N/m^2$, giving: $E=4\times10^{12} J= 1 KT$.
		This is not the total energy, since not all the energy goes into the kinetic energy of the shock, some goes to light, nuclear radiation, 
		heating and more. So an aditional factor of a few, say 4, gives an estimate for the total energy of $4 KT$.
		<br><br>
		Another physicist who used the power of estimation to estimate the energy released in the explosion, using a different 
		method and different available data. That was Taylor who in 1950 using a set of photos released to the media showing the blast wave
		expansion, conviniently giving the time elapsed and the distance scales on the photos. Taylor used the method of dimensional analysis to
		estimate this energy. In any dimensional analysis problem, one looks for dimensionless parameters constructed from the relevant physical
		parameters in the problem. The parameters he used were $E$ - total energy, $\rho$ - external density, $t$ - time elapsed, $r$ - radius of expansion.

		\[
			[E]=M L^2 S^{-2}, [\rho]=M L^{-3}, [t]=S, [r] = L
		\]

		we need to find the exponents which will give a dimensionless quantity:

		\[
			E^{a}\rho^{b}t^{c}r^{d}=const 
		\]

		\[
			M^{a}L^{2a}S^{-2a}M^bL^{-3b}S^{c}L^d=M^{a+b}L^{2a-3b+d}S^{c-2a}
		\]

		Therefore,

		\[
			\begin{split}
				a+b=0 \\
				2a-3b+d=0 \\
				c-2a=0
			\end{split}
		\]

		The answer doesn't have to be unique, since we can take the constant to any power.
		We get
		\[
			a=-b\\
			d=5b\\
			c=-2b
		\]

		taking $b=1$, gives $a=-1$, $c=-2$, $d=5$ and the dimensionless parameter is:

		\[
			const=\frac{r^5\rho}{t^2E}
		\]

		Estimating the constant to be of order unity and using a photo such as this:
<br><br>
		<center>
		<img src='images/trinity.jpg'/> 
		<br/>
		source: wikipedia.
		</center>
<br>
		taken at 25ms after the explosion with a blast radius of ~140m, he was able to get an estimation of $20KT$ which is the accurate
		value.
		So both Fermi and Taylor used estimation methods and got an order of magnitude estimation which was very close to the exact
		value, a feat which otherwise would require much more time and effort.

</div>



<script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>

</body>


</html>