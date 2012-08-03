package AI::Nerl::Model;
use Moose;
use Moose::Util::TypeConstraints;

subtype 'PositiveInt', as 'Int', where { $_ > 0 },
      message { "The number you provided, $_, was not a positive number" };

has [qw/inputs outputs/] => (
   isa => 'PositiveInt',
   is => 'ro',
   required => 1,
);
1;
