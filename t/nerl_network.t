use Test::More tests=>8;
use Modern::Perl;
use PDL;
use_ok('AI::Nerl');
use_ok('AI::Nerl::Network');
{
   my $nn = AI::Nerl::Network->new(l1=>2,l2=>2);
   isa_ok($nn,'AI::Nerl::Network');
}

#simplest 3layer: 1 in, i hidden, 1 out
#1->1,0->0
{
   my $x = pdl(0,1);
   my $y = pdl(1,0);
   my $nn = AI::Nerl::Network->new(
      l1=>1,l2=>1,l3=>1,
      lambda=>0,
      theta1=>pdl(6),
      theta2=>pdl(6),
      b1=>pdl(0),
      b2=>pdl(0),
      alpha=>100,
   );
   for(1..99){
      die $nn->run($x);
   diag($nn->run($x));
      $nn->train($x,$y,passes=>3);
      my $b1 = $nn->b2;
   diag($nn->run($x));
      die $b1 . 'foo';
      $b1->slice(0) .= 0;
   }
   diag($nn->run($x));
}

#build a few really simple nets.
#in: x1,x2
#out: x1 AND|OR|XOR x2
{
   my $x = pdl([0,0,1,1],[0,1,0,1]);
   my $AND = pdl(0,0,0,1);
   my $OR = pdl(0,1,1,1);
   my $XOR = pdl(0,1,1,0);

   my ($AND_nn,$OR_nn,$XOR_nn) = map {
      AI::Nerl::Network->new(
         l1=>2,
         l2=>2,
         l3=>1,
         alpha=>1000,
         lambda=>0,
      );
   } 1..3;
   isa_ok($XOR_nn, 'AI::Nerl::Network');
   is($XOR_nn->theta1->dim(0), 2, 'theta1 dim1 == 2');
   is($XOR_nn->theta1->dim(1), 2, 'theta1 dim2 == 2');
   is($XOR_nn->theta2->dim(0), 2, 'theta2 dim1 == 2');
   is($XOR_nn->theta2->dim(1), 1, 'theta2 dim2 == 1');
   my @Js;
   for (1..10){
      $AND_nn->train($x,$XOR, passes=>99);
   }
   warn $XOR_nn->b1;
   my $AND_output = $AND_nn->run($x);
   warn("XOR: ".$AND_output);

}
