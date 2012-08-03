#!/usr/bin/env perl
use Modern::Perl;
use FindBin qw($Bin); 
chdir $Bin;
use lib '../../lib';
use AI::Nerl;
use PDL;
use PDL::IO::FlexRaw;
use PDL::Graphics2D;
use POSIX ();
use constant E     => exp(1);

sub imag_neuron{
   my $foo = shift;
   $foo = $foo->reshape(28,28);
   $foo = normlz $foo;
   warn $foo->sum;
   imag2d $foo;
}
sub normlz{
   my $foo = shift;
   $foo = $foo - $foo->min;
   $foo /= $foo->max;
   return $foo;
}
sub imag_theta1{
   my $t1 = shift;
   my $n = $t1->dim(0);
   my $cols = 7;
   my $rows = POSIX::ceil ($n/$cols);
   my $tmp_piddle = zeroes(28*$cols,28*$rows);
   for my $i(0..$n-1){
      my $row = POSIX::floor($i/$cols);
      my $col = $i % $cols;
      my $x = $col*28;
      my $y = $row*28;
      my $slice = $tmp_piddle->slice("$x:".($x+27).",$y:".($y+27));
      $slice .= $t1->slice($i)->reshape(28,28);
   }
   imag2d normlz $tmp_piddle;
}


unless (-e "t10k-labels-idx1-ubyte.flex"){ die <<"NODATA";}
pull this data by running get_digits.sh
convert it to flexraw by running idx_to_flex.pl
NODATA


my $images = readflex('t10k-images-idx3-ubyte.flex');
$images /= 256;
my $labels = readflex('t10k-labels-idx1-ubyte.flex');

my $nerl = AI::Nerl->new(
   model => 'Perceptron3',
   model_args => {
      l2 => 10
   },
   inputs => 784,
   outputs => 10,
);

sub y_to_vectors{
   my $labels = shift;
   my $id = identity(10);
   my $y = $id->range($labels->dummy(0))->transpose;
   $y *= 2;
   $y -= 1;
   return $y->transpose
}
for(1..20){
   $nerl->train_batch(
      x => $images->slice("0:799"),
      y => y_to_vectors $labels->slice("0:799"),
   );
   $nerl->spew_cost(
      x => $images->slice("800:899"),
      y => y_to_vectors $labels->slice("800:899"),
   );
}
   print 'num correct:' . ( (
      $nerl->classify( $images->slice("1000:1499"))->flat ==
         $labels->slice("1000:1499")->flat
      )->sum
   );
imag_theta1 $nerl->model->theta1;
