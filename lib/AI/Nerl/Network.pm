package AI::Nerl::Network;
use Moose 'has', inner => { -as => 'moose_inner' };
use PDL;
use PDL::NiceSlice;
use PDL::Constants 'E';

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
has alpha => ( #learning rate
   isa => 'Num',
   is => 'rw',
   default => .36,
);
has lambda => (
   isa => 'Num',
   is => 'rw',
   default => .30,
);

sub _mk_theta1{
   my $self = shift;
   return grandom($self->l1, $self->l2) * .001;
}
sub _mk_theta2{
   my $self = shift;
   return grandom($self->l2, $self->l3) * .001;
}

   my ($labels,$delta,$out_neurons, $img,$id,$images,$pass);

sub train{
   my ($self,$x,$y) = @_;
   if ($self->scale_input){
      $x *= $self->scale_input;
   }
   my $num_examples = $x->dim(0);
   for my $pass (1..9){
      show784($self->theta1->slice(':,2'));
      my $delta1 = $self->theta1 * 0;
      my $delta2 = $self->theta2 * 0; 
      #iterate over examples :(
      for my $i (0..$num_examples-1){
         my $a1 = $x(($i));
         my $z2 = ($self->theta1 x $a1->transpose)->squeeze;
         my $a2 = sigmoid($z2);
         my $z3 = ($self->theta2 x $a2->transpose)->squeeze;
         my $a3 = sigmoid($z3);
         #my $label = $y(($i));
         
         my $d3= -($y(($i)) - $a3) * logistic($a3);# sigmoid($z3);#logistic..
         $delta2 += $d3->transpose x $a2;

         my $d2 = ($self->theta2->transpose x $d3->transpose)->squeeze * logistic($a2);#sigmoid($z2);
         $delta1 += $d2->transpose x $a1;
      }
      $self->{theta2} -= $self->alpha * ($delta2 / $num_examples + $self->theta2 * $self->lambda);
      $self->{theta1} -= $self->alpha * ($delta1 / $num_examples + $self->theta1 * $self->lambda);
      warn "theta1 wt total: ".$self->theta1->flat->sum;
      my ($cost,$numcorrect) = $self->cost($x,$y);
      warn "cost: $cost. \nclassified correctly: $numcorrect";
      
   }
   return;
   $delta /= $images->dim(0);
   $delta *= .2;
   $out_neurons -= $delta;
   if ($pass%200==0){
      warn $delta(100:104);
      warn $out_neurons(100:104);
   }  
   
}

sub cost{
   my ($self,$x,$y) = @_;
   my $n = $x->dim(1);
   if ($self->scale_input){
      $x *= $self->scale_input;
   }
   my $num_correct = 0;

   my $total_cost = 0; 
   for my $i (1..$n-1){
      my $a1 = $x(($i));
      my $z2 = ($self->theta1 x $a1->transpose)->squeeze;
      my $a2 = sigmoid($z2);
      my $z3 = ($self->theta2 x $a2->transpose)->squeeze;
      my $a3 = sigmoid($z3);
      $total_cost += ($y(($n))-$a3)->abs()->power(2,0)->sum()/2;
      warn $a3->maximum_ind . '    ' . $y(($i))->maximum_ind;;
      $num_correct++ if $a3->maximum_ind == $y(($i))->maximum_ind;
   }
   $total_cost /= $n;
   $total_cost += $self->theta1->flat->power(2,0)->sum * $self->lambda;
   $total_cost += $self->theta2->flat->power(2,0)->sum * $self->lambda;
   return ($total_cost, $num_correct);
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
sub show784{
   my $w = shift;
   $w = $w->copy;
   warn join',', $w->dims;
   $w = $w->squeeze;
   my $min = $w->minimum;
   $w -= $min;
   my $max = $w->maximum;
   $w /= $max;
   $w = $w->reshape(28,28);
   imag2d $w;
}


'$nn->train($sovietRussian)';
__END__
#my $label_targets = identity(10)->($labels);
my $id = identity(10);
$images = $images(10:11);
show784($images(0));
show784($images(1));
$labels = $labels(10:11);

my $out_neurons = grandom(28*28,10) * .01;
# http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
# http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
for my $pass(1..3){
   my $delta = $out_neurons * 0;
   for my $i(0..$images->dim(0)-1){
      my $img = $images(($i));
      my $a =  sigmoid($out_neurons x $img->transpose); #(t,10)
      #arn $out_neurons x $img->transpose if $pass > 1;; #(t,10)
      $a = $a((0));
      my $label = $labels(($i));
      my $d= $id(($label)) - $a;
      $d = -$d * $a * (1-$a); #(t,10)
      $delta += $d->transpose x $img;
      if (rand() < 1.002){
         warn $d;
         warn $a;
         warn "$label -> " . $a->maximum_ind;
         say "\n"x 2;
      }
      if ($pass%250==0 and $i<5){
         warn $a;
         warn $d;
         warn "$label -> " . $a->maximum_ind;
      }
   }
   $delta /= $images->dim(0);
   $delta *= .2;
   $out_neurons -= $delta;
   if ($pass%200==0){
      warn $delta(100:104);
      warn $out_neurons(100:104);
   }  
   show784($delta(:,0));
   show784($delta(:,6));
   show784($delta(:,4));
}
#die join (',',$nncost->dims);
use PDL::Graphics2D;
sub show784{
   my $w = shift;
   $w = $w->squeeze;
   my $min = $w->minimum;
   $w -= $min;
   my $max = $w->maximum;
   $w /= $max;
   $w = $w->reshape(28,28);
   imag2d $w;
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
