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

extends 'AI::Nerl::Model';

# nice 3-layer perceptron.
# immutable attributes. try cloning or something
#  if you want to change something.


# these are inherited from model:
# subtype 'PositiveInt'
# has [qw/inputs outputs/] 

#hidden size
has _l2_size => (
   init_arg => 'l2',
   is => 'ro',
   isa => 'PositiveInt',
   required => 1,
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
   return $self->theta0->dim(0);
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
#      my $exp = exp($in*2);
#      return (($exp-1)/($exp+1));
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
   my $ins = $self->inputs;
   my $outs = $self->outputs;
   my $l2 = $self->_l2_size;
   $self->_privately_write_theta1(grandom($l2,$ins) * .0001);
   $self->_privately_write_theta2(grandom($outs,$l2) * .001);
   $self->_privately_write_b1(grandom($l2));
   $self->_privately_write_b2(grandom($outs));
};

#a dimensional transform from $inputs to $outputs
sub run{
   my ($self, $x) = @_; 
}
#a dimensional transform from $inputs to 1
sub classify{
   my ($self, $x) = @_; 
}

sub train_batch{
   my $self = shift;
   $self->train(@_) for 1..1;
}

sub spew_cost{
   my ($self, %args) = @_; 
   my $x = $args{x};
   my $y = $args{y};
   my $theta1 = $self->theta1;
   my $theta2 = $self->theta2;
   my $b1 = $self->b1;
   my $b2 = $self->b2;

   my $z2 = $x->transpose x $theta1;
   $z2 += $b1;
   my $a2 = $self->_act->($z2);

   my $z3 = $a2 x $theta2;
   $z3 += $b2;
   my $a3 = $self->_act->($z3);
   #warn $a2->slice(":,3");

   my $J = .5 * (sum(($a3 - $y)**2)) / $x->dim(1);;
   print "COST: $J \n";
}

#modify the b's,thetas
sub train{
   my ($self, %args) = @_; 
   my $x = $args{x};
   my $y = $args{y};

   my $passes = 1;
   my $alpha = .2;
   my $lambda = .00;

   my $n = $x->dim(0);

   my $theta1 = $self->theta1;
   my $theta2 = $self->theta2;
   my $b1 = $self->b1;
   my $b2 = $self->b2;

   # I sorta care that the input vectors remain vertical.
   # so for 100 inputs, dim(1)==100
   # dim(0) == bias
   my $z2 = $x->transpose x $theta1;
   $z2 += $b1;
   my $a2 = $self->_act->($z2);

   my $z3 = $a2 x $theta2;
   $z3 += $b2;
   my $a3 = $self->_act->($z3);

   #die $a3->dims; #(cats,n)
   #so far so good.
   #die $z3->slice("0:5,0:5");

   my $d3 = -($y-$a3)*$self->_act_deriv->($z3);
#   warn $self->_act_deriv->($z3)->slice("0:5,0:3");;
   # warn $x->slice("10:18,10:18");;
   my $d2 = ($d3 x $theta2->transpose) * $self->_act_deriv->($z2);

   my $delta2 = $a2->transpose x $d3;
   my $delta1 = $x x $d2;
   my $deltab2 = $d3->transpose->sumover->flat;
   my $deltab1 = $d2->transpose->sumover->flat;
   
   my $difft1 = $alpha * (($delta1/$n) + ($theta1 * $lambda));
   $self->theta1->inplace->minus($difft1,0);

   my $difft2 = $alpha * ($delta2/$n + $lambda*$theta2);
   $self->theta2->inplace->minus($difft2,0);
   
   my $diffb2 = ($alpha/$n)*$d3;
   $self->b2->inplace->minus($diffb2,0);
   my $diffb1 = ($alpha/$n)*$d2;
   $self->b1->inplace->minus($diffb1,0);

   return;
   #iterate over examples :(
=pod
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
   $self->{theta2} -= $alpha * ($delta2 / $num_examples + $theta2 * $lambda);
   $self->{theta1} -= $alpha * ($delta1 / $num_examples + $theta1 * $lambda);
   $self->{b1} -= $alpha * $deltab1 / $num_examples;
   $self->{b2} -= $alpha * $deltab2 / $num_examples;
=cut
}

1;


