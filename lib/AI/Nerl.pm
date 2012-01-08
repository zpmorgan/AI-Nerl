package AI::Nerl;
use Moose 'has',inner => { -as => 'moose_inner' };
use PDL;
use AI::Nerl::Network;

#A Nerl is a mechanism to build neural networks?
#Give it training,test, and cv data?
#it settles on a learning rate and stuff?
#or maybe it's also a language for guided training?
#or maybe a visual gui thing?
#Not exactly sure. Maybe I'm tinkering with forces better left alone.
#That's a great excuse for failing horribly.

has scale_input => (
   is => 'ro',
   isa => 'Num',
   required => 0,
   default => 0,
);
has l2 => ( #hidden layer.
   is => 'ro',
   isa => 'Num',
   default => 30,
);

has [qw/ train_x 
         train_y /] => (
   is => 'ro',
   isa => 'PDL',
);
has [qw/ test_x cv_x
         test_y cv_y /] => (
   is => 'ro',
   isa => 'PDL',
   required => 0,
);

has network => (
   required=>0,
   is => 'rw',
   isa => 'AI::Nerl::Network',
);

sub build_network{
   my $self = shift;
   my $nn = AI::Nerl::Network->new(
      l1 => $self->train_x->dim(1),
      l2 => $self->l2,
      l3 => $self->train_y->dim(1),
      scale_input => $self->scale_input,
   );
   $nn->train($self->train_x, $self->train_y);
   $self->network($nn);
}



'a neural network has your dog.';
