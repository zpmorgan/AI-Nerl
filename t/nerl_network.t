use Test::More tests=>8;
use Modern::Perl;
use PDL;
use_ok('AI::Nerl');
use_ok('AI::Nerl::Network');
{
   my $nn = AI::Nerl::Network->new(l1=>2,l2=>2);
   isa_ok($nn,'AI::Nerl::Network');
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
         l2=>3,
         l3=>1,
      );
   } 1..3;
   isa_ok($XOR_nn, 'AI::Nerl::Network');
   is($XOR_nn->theta1->dim(0), 2, 'theta1 dim1 == 2');
   is($XOR_nn->theta1->dim(1), 3, 'theta1 dim2 == 3');
   is($XOR_nn->theta2->dim(0), 3, 'theta1 dim1 == 3');
   is($XOR_nn->theta2->dim(1), 1, 'theta1 dim2 == 1');

   $AND_nn->train($x,$AND, passes=>111);
   my $AND_output = $AND_nn->run($x);
   diag($AND_output);

}
