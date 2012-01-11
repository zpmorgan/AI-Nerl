package AI::Nerl::Network;
use Moose 'has', inner => { -as => 'moose_inner' };
use PDL;
use PDL::NiceSlice;
use PDL::Constants 'E';

my $DEBUG=0;

# http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
# http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf


# Simple nn with 1 hidden layer
# train with $nn->train(data,labels);
has scale_input => (
   is => 'ro',
   required => 0,
   isa => 'Num',
   default => 0,
);

# number of input,hidden,output neurons
has [qw/ l1 l2 l3 /] => (
   is => 'ro',
   isa => 'Int',
);

has theta1 => (
   is => 'ro',
   isa => 'PDL',
   lazy => 1,
   builder => '_mk_theta1',
);
has theta2 => (
   is => 'ro',
   isa => 'PDL',
   lazy => 1,
   builder => '_mk_theta2',
);

has b1 => (
   is => 'ro',
   isa => 'PDL',
   lazy => 1,
   builder => '_mk_b1',
);
has b2 => (
   is => 'ro',
   isa => 'PDL',
   lazy => 1,
   builder => '_mk_b2',
);
has alpha => ( #learning rate
   isa => 'Num',
   is => 'rw',
   default => .3,
);
has lambda => (
   isa => 'Num',
   is => 'rw',
   default => .01,
);

sub _mk_theta1{
   my $self = shift;
   return grandom($self->l1, $self->l2) * .001;
}
sub _mk_theta2{
   my $self = shift;
   return grandom($self->l2, $self->l3) * .001;
}
sub _mk_b1{
   my $self = shift;
   return grandom($self->l2) * .001;
}
sub _mk_b2{
   my $self = shift;
   return grandom($self->l3) * .001;
}


sub train{
   my ($self,$x,$y, %params) = @_;
   $x->sever();
   my $passes = $params{passes} // 10;

   if ($self->scale_input){
      $x *= $self->scale_input;
   }
   my $num_examples = $x->dim(0);

   for my $pass (1..$passes){
#      warn 'blah:'. $self->theta1->slice(':,2')->flat->sum;
      show784($self->theta1->slice(':,0')) if $pass%30==29 and $DEBUG;
      my $delta1 = $self->theta1->copy * 0;
      my $delta2 = $self->theta2->copy * 0;
      my $deltab1 = $self->b1->copy * 0;
      my $deltab2 = $self->b2->copy * 0;

      #iterate over examples :(
      for my $i (0..$num_examples-1){
         my $a1 = $x(($i));
         my $z2 = ($self->theta1 x $a1->transpose)->squeeze;
         $z2 += $self->b1; #add bias.
         my $a2 = $z2->tanh;
         my $z3 = ($self->theta2 x $a2->transpose)->squeeze;
         $z3 += $self->b2; #add bias.
         my $a3 = $z3->tanh;
         # warn $y(($i)) - $a3;
         my $d3= -($y(($i)) - $a3) * tanhxderivative($a3);
         #warn $d3;
         $delta2 += $d3->transpose x $a2;
         my $d2 = ($self->theta2->transpose x $d3->transpose)->squeeze * tanhxderivative($a2);
         $delta1 += $d2->transpose x $a1;
         #warn $delta2(4);
         $deltab1 += $d2;
         $deltab2 += $d3;

         if($DEBUG==1){
            warn "z2: $z2\n$z3: $z3\n";
            warn "d3:$d3\n";
         }
      }
      #warn $deltab1;
      if($DEBUG==1){
         warn "theta1: ". $self->theta1;#/ $num_examples;
         warn "theta2: ". $self->theta2;
         warn "delta1: $delta1\n";
         warn "delta2: $delta2\n";
      }
      $self->{theta2} -= $self->alpha * ($delta2 / $num_examples + $self->theta2 * $self->lambda);
      $self->{theta1} -= $self->alpha * ($delta1 / $num_examples + $self->theta1 * $self->lambda);
      $self->{b1} -= $self->alpha * $deltab1 / $num_examples;
      $self->{b2} -= $self->alpha * $deltab2 / $num_examples;
   }
}

sub run{
   my ($self,$x) = @_;
   $x->sever();
   if ($self->scale_input){
      $x *= $self->scale_input;
   }

   $x = $x->transpose if $self->l1 != $x->dim(1);
   my $y = $self->theta1 x $x;
   $y += $self->b1->transpose;
   $y->inplace()->tanh;# = tanhx($y);

   $y = $self->theta2 x $y;
   $y += $self->b2->transpose;
   $y->inplace()->tanh();# = tanhx($y);
   return $y;
}

sub cost{
   my ($self,$x,$y) = @_;
   $x->sever();# = $x->copy();
   my $n = $x->dim(0);
   if ($self->scale_input){
      $x *= $self->scale_input;
   }
   my $num_correct = 0;
   #die join(',',$x->dims) .',,,'. join(',',$y->dims);
   my $total_cost = 0; 
   for my $i (0..$n-1){
      my $a1 = $x(($i));
      my $z2 = ($self->theta1 x $a1->transpose)->squeeze;
      $z2 += $self->b1;
      my $a2 = $z2->tanh();
      my $z3 = ($self->theta2 x $a2->transpose)->squeeze;
      $z3 += $self->b2;
      my $a3 = $z3->tanh;
      $total_cost += ($y(($i))-$a3)->abs()->power(2,0)->sum()/2;
      #warn $a3->maximum_ind . '    ' . $y(($i))->maximum_ind;;
      $num_correct++ if $a3->maximum_ind == $y(($i))->maximum_ind;
   }
   $total_cost /= $n;
   $total_cost += $self->theta1->flat->power(2,0)->sum * $self->lambda;
   $total_cost += $self->theta2->flat->power(2,0)->sum * $self->lambda;
   return ($total_cost, $num_correct);
}

sub tanhx{ #don't use this. pdl has $pdl->tanh which can be used in place.
   my $foo = shift;
   my $p = E**$foo;
   my $n = E**-$foo;
   return (($p-$n)/($p+$n));
}
sub tanhxderivative{ #use: tanhxderivative($pdl->tanh()). save time by finding tanh first.
   my $tanhx = shift;
   return (1 - $tanhx**2);
}

sub sigmoid{
   my $foo = shift;
   return 1/(1+E**-$foo);
}

sub logistic{
   #find sigmoid before calling this.
   #grad=logistic(sigmoid(foo))
   my $foo = shift;
   return $foo * (1-$foo);
}

use PDL::Graphics2D;
#display 28x28 grayscale pdl.
sub show784{
   my $w = shift;
   $w = $w->copy;
   #warn join',', $w->dims;
   $w = $w->squeeze;
   my $min = $w->minimum;
   $w -= $min;
   my $max = $w->maximum;
   $w /= $max;
   $w = $w->reshape(28,28);
   imag2d $w;
}

sub show_neuron{
   my $self = shift;
   my $n = shift // 0;
   my $x = shift || 28;
   my $y = shift || 28;
   my $w = $self->theta1->slice(":,$n")->copy;
   $w = $w->squeeze;
   my $min = $w->minimum;
   $w -= $min;
   my $max = $w->maximum;
   $w /= $max;
   $w = $w->reshape($x,$y);
   imag2d $w;
}

'$nn->train($sovietRussian)';

