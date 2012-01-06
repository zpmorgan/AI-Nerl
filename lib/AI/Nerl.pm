package AI::Nerl;
use Moose;

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
);

has [qw/ test_data   train_data   cv_data
         test_labels train_labels cv_labels /] => (
   is => 'ro',
   isa => 'PDL',
);

has network => (
   required=>0,
   is => 'rw',
   isa => 'AI::Nerl::Network',
);

sub build_network{
   my $self = shift;
   my $nn = AI::Nerl::Network->new(
      l1 => $self->train_data->dim(1),
      l2 => 30,
      l3 => $self->train_labels->dim(0),
      scale_input => $self->scale_input,
   );
   $self->network($nn);
}



'a neural network has your dog.';
