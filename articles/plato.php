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
<title>Plato solids</title>
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

		<h1 class='special'>Platonic solids</h1>
For some reason, our mind has evolved in such a way as to deeply and inherently appreciate symmetry. We seek it everywhere
and identify it automatically. Nature seems also to appreciate symmetry, as evident in living creatures, plants and some 
inanimate objects. A fact that may be understood from principles of efficiency and energy minimization under certain constraints.
Even more profoundly, the fundamental laws of nature exhibit certain symmetries, a fact that has become a central and 
guiding idea in physics when formulating new theories. The central symmetry that must be fulfilled 
by all possible theories of nature is symmetry under the Poincare Group. This group contains various elements which correspond to 
different transformation that can be applied on an equation or an action without changing their content or resulting 
equations of motion and solutions. The Poincare group is of dimension 10 which means that 10 parameters are required to 
generate any element of the group, these contain the 6 types of transformations of the Lorentz group (3 boosts and 3 rotations) 
and 4 translations in time and space. One of the most fundamental symmetries of nature is called CPT symmetry which states that 
all processes are symmetric under the combination of time reversal followed by spatial inversion (parity) and followed by 
complex conjugation (particle$\rightarrow$anti-particle), however there could be processes which are not symmetric under one of the 
mentioned operations. Interestingly these three transformation result in a joined parity transformation of space and time, which
can be obtained as a combination of three boosts with an imaginary rapidity ($\chi=\pm i\pi$).
In particle physics, the elementary particles and forces (except for gravity) are described in terms of symmetry groups which
are essentially rotations in the space of the elementary particles, and the generators of those groups are the force carriers. The
standard model of particle physics describes the strong force, weak force and electromagnetism as a direct product of three groups 
of different dimensions: $SU(3)\otimes SU(2)\otimes U(1)$.
<br>
<br>

The presence of symmetry in physics is fundamental and broad in all areas of physics. 
Here I want to focus on a much simpler example than discussed before, after viewing a <a href='https://www.youtube.com/watch?v=rhdx2-E3nOQ' target='_blank'>lecture</a> by Nobel laureate
Frank Wilczek. Therefore, some of the ideas are based on that talk.

<br>
<br>

Perhaps one of the first examples of relating symmetry and the real world is the Platonic solids named after the Greek Philosopher Plato. Plato claimed
based just on elegance and beauty, that the 5 solids are related to the 4+1 elements of the Universe (fire, water, earth, air, universe).
The Platonic solids are beautiful 3-dimensional objects which show a large degree of symmetry. 
The solids belong to the set of convex polyhedra (composed of flat faces, straight edges and vertices, and are closed meaning that the line 
between any two points within the solid is contained within it as well). Their beauty arises from their face being 
regular polygons (equal sides and equal angles), with the same number of polygons meeting at every vertex.

These are basically the natural generalization of regular polygons in 2D. However, unlike in 2D where one can make an infinite number
of regular polygons by increasing the number of edges (reaching a circle in the limit of infinite edges), once you go to 3 dimensions,
suddenly there are only 5 such objects, which is much less than infinity, and therefore surprising. Also, in 3D in analogy to 2D the ultimate 
symmetrical shape is of course the sphere. So why are there only 5 such objects and what are they?

To answer these questions we first need to understand a fundamental formula of planar graphs.  <br>
<span style='font-weight:bold'> Euler's Formula </span> states that for a finite, connected, planar graph G=(V,E) (with vertices V and edges E) with 
$v\ge 1$ vertices, $e$ edges and $f$ faces, the following holds: 
\begin{equation}
	v-e+f=2
\end{equation}

The proof of this formula uses induction on the number of edges $e$: <br>

base case $e=0$: $v=1$, $f=1$ and we have $v-e+f=1-0+1=2$ as required. <br>
induction step: assume that for any planar connected graph with $v$ vertices, $e\le n$ edges and $f$ regions: $v-e+f=2$.
Now consider a graph with v vertices, $n+1$ edges and $f$ regions.
There are two cases, if G is a tree then the number of edges is one less than the number of vertices $e=v-1$, and since there are no
circles in a tree, $f=1$ (the outer region), and we get $v-e+f=v-v+1+1=2$ as required.
<br>
In the second case, $G$ is not a tree and therefore there is a cycle C. consider the sub graph of G without one of the edges of the circle 
$G'=(V,E\backslash e_1)$. $G'$ is still connected and planar, but has one fewer regions $f=f-1$, and $e=n$. Since $e=n$ it fits in the inductive
assumption and we have for $G'$: $v-e+f=v-n+(f-1)=2$ this can also be written as $v-e+f=v-(n+1)+f=2$, which implies that
for the graph $G$ with $v$ vertices, $n+1$ edges and f regions the formula also applies, completing the proof by induction, 
and thus the formula applies for all $e\ge0$. Q.E.D

<br/><br/>
How does this relate to polyhedra and the Platonic solids? The crucial connection is the fact that graphs drawn on a sphere 
follow the same Euler formula which we proved for the case of a plane.

Therefore any convex polyhedron can be projected with its vertices and edges on the sphere and the sphere can then be punctured 
and turned into a 2D plane. This means that any planar graph (whose definition requires having no edges crossing) can be equivalently
drown on the surface of a sphere. Mathematically this is stated by saying that the set of all planar graphs is equal to the set
of all graphs with a genus of zero, $g=0$.
The genus is the subscript of the first surface on which the graph can be drawn without edge-crossing. For example, $S_0$ is the sphere,
$S_1$ is the torus. 

Now we can move to consider the allowed number of edges, vertices and faces to fulfill the requirements for the Platonic solids.
We denote the number of edges of the regular polygons with $s\ge 3$, and the degree of each vertex by $k\ge 3$ since the angles
of the polygons are less than $180$ degrees. <br>

Next we make the following observations: <br>
&#149; $2e=vk$, this follows from the fact that twice the number of edges equals the sum of the degrees of the vertices, since we count
twice every edge from the two vertices that are connected by an edge. since we have $v$ edges and each with k vertices, the right hand side
is $vk$. <br>
&#149; $2e=sf$, this follows from the fact that each face has $s$ edges, and each edge separates two regions, therefore we count the edges
twice by counting the number of faces times the edges of each face.<br><br>

Plugging these into Euler's formula we have:

\[

\frac{2}{k}e-e+\frac{2}{s}e=2 \\

\frac{2}{k}+\frac{2}{s}-1=\frac{2}{e} \\

\frac{2s+2k-ks}{ks}=\frac{2}{e} \\

e=\frac{2ks}{2s+2k-ks}

\]

\begin{equation}
e=\frac{2ks}{4-(k-2)(s-2)}
\end{equation}

from this we have using the previous relations:

\begin{equation}
v=\frac{2s}{4-(k-2)(s-2)}
\end{equation}


\begin{equation}
f=\frac{2k}{4-(k-2)(s-2)}
\end{equation}


<br>

From $(2)$, $(3)$, and $(4)$ we can construct different polyhedra by plugging values of $k$ (vertices degree) and $s$ (polygon edges). 
The possible values are: $(k,s)=(3,3),(3,4),(4,3),(3,5),(5,3)$, and these are the only possible values from which we get
positive $e,v,f$. <br>
These correspond to the following   : <br>
$(k,s)=(3,3)$ : Tetrahedron <br>
$(k,s)=(3,4)$ : Cube <br>
$(k,s)=(4,3)$ : Octahedron <br>
$(k,s)=(3,5)$ : Dodecahedron <br>
$(k,s)=(5,3)$ : Icosahedron <br>

<br><br>
<center>
<table>
<tr>
<td style='text-align:center;'><img src='images/plato1.png' width='70%' /></td>
<td style='text-align:center;'><img src='images/plato2.png' width='70%' /></td>
<td style='text-align:center;'><img src='images/plato3.png' width='70%' /></td>
<td style='text-align:center;'><img src='images/plato4.png' width='70%' /></td>
<td style='text-align:center;'><img src='images/plato5.png' width='70%' /></td>
</tr>
<tr>
<td style='text-align:center;'>Tetrahedron</td>
<td style='text-align:center;'>Cube</td>
<td style='text-align:center;'>Octahedron</td>
<td style='text-align:center;'>Dodecahedron</td>
<td style='text-align:center;'>Icosahedron</td>
</tr>
</table>
</center>

</div>


</body>


</html>

