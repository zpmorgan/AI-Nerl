use Test::More tests=>2;
use Modern::Perl;
use PDL;
use_ok('AI::Nerl');
{
   my $nerl = AI::Nerl->new();
   isa_ok($nerl,'AI::Nerl');
}

#task: mod 3
#in: 8 bits from (n=0..255);
#out: 1 output: (n%3 != 0)
my $x = map{split '',sprintf("%b",$_)} 0..255;
$x = pdl($x)->transpose;
my $y = pdl map{$_%3 ? 1 : 0} 0..255;
$y = identity(3)->range($y->transpose);

my $nerl = AI::Nerl->new(
   train_x => $x,
   train_y => $y,
);

#$nerl->init_network();
$nerl->build_network();



