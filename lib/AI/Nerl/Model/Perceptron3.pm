package AI::Nerl::Model::Perceptron3;
use Moose qw'extends with has';
use Moose::Util::TypeConstraints (qw/enum subtype as message/, where=>{-as, 'whereas'});
use PDL;
#use PDL::NiceSlice;
#use PDL::Constants 'E';
use constant E => exp(1);
#use File::Path;
use MooseX::Storage;

with Storage('format' => 'JSON', 'io' => 'File');

our $VERSION=0.01;

extends 'AI::Nerl::Model';

# nice 3-layer perceptron.
# immutable attributes. try cloning or something
#  if you want to change something.


# these are inherited from model:
# subtype 'PositiveInt'
# has [qw/inputs outputs/] 

#hidden size. argument.
has _l2_size => (
   init_arg => 'l2',
   is => 'ro',
   isa => 'PositiveInt',
   required => 0,
#   traits => ['DoNotSerialize'],
);

for(qw/theta1 theta2 b1 b2/){
   has $_ => (
   isa => 'PDL',
   is => 'ro',
   traits => ['DoNotSerialize'],
   writer => "_privately_write_$_",
   );
}

sub l1{
   my $self = shift;
   return $self->theta1->dim(1);
}
sub l2{
   my $self = shift;
   return $self->theta1->dim(0);
}
sub l3{
   my $self = shift;
   return $self->theta2->dim(0);
}

enum 'ActivationFunction', [qw/signoid tanh/];
has activation_function => (
   isa => 'ActivationFunction',
   is => 'ro',
   default => 'tanh',
);

has '_act' => (
   isa => 'CodeRef',
   is => 'ro',
   lazy => 1,
   traits => ['DoNotSerialize'],
   builder => '_build_act_sub',
);
has '_act_deriv' => (
   isa => 'CodeRef',
   is => 'ro',
   lazy => 1,
   traits => ['DoNotSerialize'],
   builder => '_build_act_deriv_sub',
);

sub _build_act_sub{
   my $self = shift;
   die if $self->activation_function ne 'tanh';
   return sub{
      my $in = shift;
      return $in->tanh;
   };
}
sub _build_act_deriv_sub{
   my $self = shift;
   die if $self->activation_function ne 'tanh';
   return sub{
      my $in = shift;
      return (1 - ($in->tanh)**2);
   };
}

# this initializes theta and bias piddles.
# this should also load from some crapfile if saved.
sub BUILD{
   my $self = shift;
   # are these piddles provided?
   return if defined $self->theta1;

   my $ins = $self->inputs;
   my $outs = $self->outputs;
   my $l2 = $self->_l2_size;
   die 'need an l2 size' unless $l2;
   $self->_privately_write_theta1(grandom($l2,$ins) * .00001);
   $self->_privately_write_theta2(grandom($outs,$l2) * .00001);
   $self->_privately_write_b1(grandom($l2)*.0001);
   $self->_privately_write_b2(grandom($outs)*.0001);
};

#a dimensional transform from $inputs to $outputs
sub run{
   my ($self, $x) = @_; 
   my $theta1 = $self->theta1;
   my $theta2 = $self->theta2;
   my $b1 = $self->b1;
   my $b2 = $self->b2;

   my $z2 = $theta1->transpose x $x;
   $z2->transpose->inplace->plus($b1,0);
   my $a2 = $self->_act->($z2);

   my $z3 = $theta2->transpose x $a2;
   $z3->transpose->inplace->plus($b2,0);
   my $a3 = $self->_act->($z3);

   return $a3;
}
#a dimensional transform from $inputs to 1
sub classify{
   my ($self, $x) = @_; 
   my $a3 = $self->run($x);
   my $maxes = $a3->transpose->maximum_ind;;
   return $maxes;
   die $maxes->slice("5:15");
   die $maxes->dims;
   die $a3->slice("3:4");
   die $a3->dims;
}

sub train_batch{
   my $self = shift;
   $self->train(@_) for 1..1;
}

sub spew_cost{
   my ($self, %args) = @_; 
   my $y = $args{y} // die 'need y';;
   my $x = $args{x} // die 'need x';
   my $lambda = $args{lambda} // .04;
   #my $alpha = $args{alpha} // .25;
   my $theta1 = $self->theta1;
   my $theta2 = $self->theta2;
   my $b1 = $self->b1;
   my $b2 = $self->b2;

   my $z2 = $theta1->transpose x $x;
   $z2->transpose->inplace->plus($b1,0);
   my $a2 = $self->_act->($z2);

   my $z3 = $theta2->transpose x $a2;
   $z3->transpose->inplace->plus($b2,0);
   my $a3 = $self->_act->($z3);
   
   my $J = (.5/$x->dim(0)) * sum(($a3 - $y)**2);;
   $J += ($lambda/2) * (($theta1**2)->sum + ($theta2**2)->sum);
   print "COST: $J \n";
   my $maxes = $a3->transpose->maximum_ind;;
   my $labels = $y->transpose->maximum_ind;
   # die $theta2;
   #die $a3;
   print "num correct(of ".($x->dim(0))."): ".(($labels==$maxes)->sum)."\n";
}

#modify the b's,thetas
sub train{
   my ($self, %args) = @_; 
   my $x = $args{x}; #dims: (cases,inputs)
   my $y = $args{y}; #(cases,outputs)

   my $alpha = $args{alpha} // .12;
   my $lambda = $args{lambda} // .04;

   my $n = $x->dim(0);

   my $theta1 = $self->theta1;
   my $theta2 = $self->theta2;
   my $b1 = $self->b1;
   my $b2 = $self->b2;

   # I sorta care that the input vectors remain vertical.
   # so for 100 inputs, dim(1)==100
   # dim(0) == bias
   my $z2 = $theta1->transpose x $x;
   $z2->transpose->inplace->plus($b1,0);
   my $a2 = $self->_act->($z2);

   my $z3 = $theta2->transpose x $a2;
   $z3->transpose->inplace->plus($b2,0);
   my $a3 = $self->_act->($z3);
   #so far so good.

   my $d3 = -($y-$a3)*$self->_act_deriv->($z3);
   my $d2 = ($theta2 x $d3) * $self->_act_deriv->($z2);

   my $delta2 = $a2 x $d3->transpose;
   my $delta1 = $x x $d2->transpose;
   my $deltab2 = $d3->sumover->flat;
   my $deltab1 = $d2->sumover->flat;

   my $difft1 = $alpha * (($delta1/$n) + ($lambda*$theta1));
   $self->theta1->inplace->minus($difft1->copy,0);

   my $difft2 = $alpha * (($delta2/$n) + ($lambda*$theta2));
   $self->theta2->inplace->minus($difft2->copy,0);
   
   my $diffb2 = ($alpha/$n)*$deltab2;
   $self->b2->inplace->minus($diffb2->copy,0);
   my $diffb1 = ($alpha/$n)*$deltab1;
   $self->b1->inplace->minus($diffb1->copy,0);
   #warn "difft2: $difft2"; 
   #warn "diffb2: $diffb2"; 

   return;
}

use PDL::IO::FlexRaw;
use File::Slurp;
use JSON;
sub save_to_dir{
   my ($self,$dir, %args) = @_;
   die 'mustbeclasspathdir' unless $dir->isa ('Path::Class::Dir');
   if (-e $dir){ #why overwrite? nerls should be saved directly, not models
      die "direxists $dir" unless $args{overwrite};
      $dir->rmtree;
   }
   $dir->mkpath;
   my $frozen = $self->freeze;
   write_file($dir->file('perceptron3.json')->stringify, $frozen);
   #theta1,theta2,b1,b2
   $PDL::IO::FlexRaw::writeflexhdr = 1;
   writeflex($dir->file('piddles.flex')->stringify,
      $self->theta1,$self->theta2,$self->b1,$self->b2);
}
sub load_from_dir{
   my $class = shift;
   my $dir = shift;
   die 'mustbeclasspathdir' unless $dir->isa ('Path::Class::Dir');
   die 'dirNonExistant '.$dir unless -e $dir;
   my $frozen = read_file($dir->file('perceptron3.json'));
   my ($theta1,$theta2,$b1,$b2) = readflex($dir->file('piddles.flex')->stringify);

   my $from_json = from_json($frozen);
   my $self = AI::Nerl::Model::Perceptron3->unpack($from_json, 
      inject => {
         theta1 => $theta1, theta2 => $theta2,
         b1 => $b1, b2 => $b2,
      });
   return $self;
}


sub export_c{
   my $self = shift;
   my $hdr = "#include <math.h>\n\n";
   my $externs = '';
   $externs .= "extern float*  x;\n";
   $externs .= "extern float* a2;\n";
   $externs .= "extern float* a3;\n\n";
   my $globals = '';
   $globals .= "float  x[".$self->l1."];\n";
   $globals .= "float a2[".$self->l2."];\n";
   $globals .= "float a3[".$self->l3."];\n\n";

   my @functions;
   
   for my $i (0..$self->l2-1){
      my @func_lines = "float do_l2_n$i (){";
      push @func_lines, "  float sum = 0;";
      #for my $j (0..$self->l1-1){
         #push @func_lines, "  sum += x[$j] * ".$self->theta1->at($i,$j) . ';';
      #}
      my @elems;
      for my $j (0..$self->l1-1){
         next if rand() > .001;
         push @elems, " (x[$j] *1000* ".$self->theta1->at($i,$j) .') ';
      }
      push @func_lines, 'sum = '.join('+',@elems) . ';';
      push @func_lines, "  float act = tanh(sum);";
      push @func_lines, "  return act;";
      push @func_lines, "}";
      $hdr .= "float do_l2_n$i(); \n";
      push @functions, join ("\n",@func_lines);
   }
   for my $i (0..$self->l3-1){
      my @func_lines = "float do_l3_n$i (){";
      push @func_lines, "  float sum = 0;";
      #for my $j (0..$self->l2-1){
      #   push @func_lines, "  sum += a2[$j] * ".$self->theta2->at($i,$j) . ';';
      #}
      my @elems;
      for my $j (0..$self->l2-1){
         push @elems, " (a2[$j] * ".$self->theta2->at($i,$j) .') ';
      }
      push @func_lines, 'sum = '. join('+',@elems) .';';
      push @func_lines, "  float act = tanh(sum);";
      push @func_lines, "  return act;";
      push @func_lines, "}";
      push @functions, join ("\n",@func_lines);
      $hdr .= "float do_l3_n$i(); \n";
   }
   {
      my @func_lines = "void do_l2(){";
      push @func_lines, map {"  a2[$_]=do_l2_n$_(x);"} 0..$self->l2-1;
      push @func_lines, "}";
      push @functions, join ("\n",@func_lines);
      $hdr .= "void float_do_l2();\n";
   }
   {
      my @func_lines = "void do_l3(){";
      push @func_lines, map {"  a3[$_]=do_l3_n$_(a2);"} 0..$self->l3-1;
      push @func_lines, "}";
      push @functions, join ("\n",@func_lines);
      $hdr .= "void float_do_l3();\n";
   }
   { #assume input is in rgb, 0 to 255..
      #since this is main-ish, i'll put globals here I guess.
      my @func_lines;# = $globals;
      push @func_lines, "classify_from_rgba32(char* x_in) {";
      #push @func_lines, map {"  x[$_] = (float)(x_in[$_]) / 255 ;"} 0..$self->l1-1;

      push @func_lines, "  int xwlsh;";
      push @func_lines, "  for(xwlsh=0; xwlsh<".$self->l1."; xwlsh++){";
      push @func_lines, "    x[xwlsh] = (float)(x_in[xwlsh]) / 255 ;";
      push @func_lines, "  }";

      push @func_lines, "  do_l2();";
      push @func_lines, "  do_l3();\n";

      push @func_lines, "  int i;";
      push @func_lines, "  int max_i;";
      push @func_lines, "  float max = -2;";
      push @func_lines, "  for(i=0;i<".$self->l3.";i++){";
      push @func_lines, "    if(a3[i] > max){";
      push @func_lines, "      max = a3[i];";
      push @func_lines, "      max_i = i;";
      push @func_lines, "    }";
      push @func_lines, "  }";
      push @func_lines, "  return max_i;";
      push @func_lines, "}";

      push @functions, join ("\n",@func_lines);
   }
   #my $txt = $hdrs . join("\n\n",@functions);
   return {header => $hdr, functions => \@functions, externs=>$externs, globals=>$globals};
   #return $txt;
}



'$nn->train($soviet_russian);'

