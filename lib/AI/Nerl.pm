package AI::Nerl;
use Moose (qw'around has with' ,inner => { -as => 'moose_inner' });
use PDL;
use AI::Nerl::Network;
use File::Slurp;
use JSON;

# ABSTRACT: generalized machine learning?

# main_module

our $VERSION = .04;

# A Nerl is an umbrella class to support different ML modules.
# train it with batch or online modes. trainers provided.
# I guess it settles on different parameters?
# This is pretty much a reboot.
# Importer for earler version?


=head1 AI::Nerl - Generalized machine learning. only perceptrons for now.

=head1 SYNOPSIS

Check out L<AI::Nerl::Model::Perceptron3>; This module is in an early stage.
Future plans are to support modular neural networks, rbf networks, and SVMs
of some sort.

Basically, each nerl has a model (like such as perceptron3) & maybe a trainer 
(like such as AI::Model::Perceptron3::BatchTrainer?).

Models should implement run, maybe provide changesets from training.
They are initialized with their own set of arguments provided to nerl's constructor.

Nerl has inputs & outputs. all training(batch & online) ought to
be directed through the nerl, which selects an appropriate trainer.

=head1 AUTHOR

Zach Morgan

=cut


# how to handle resizing?
# maybe generate a new model & copy data over.
has [qw/inputs outputs/] => (
   is => 'ro',
   isa => 'Int',
);

has _model_init_args => (
   init_arg => 'model_args',
   is => 'ro',
   isa => 'HashRef',
   default => sub{{}},
);
has _specified_model_type => (
   #init_arg => 'model', #munged in BUILDARGS from 'model'
   is => 'ro',
   isa => 'Str',
   required => 1,
);
has model => (
   is => 'rw',
   isa => 'AI::Nerl::Model',
   builder => '_build_model',
   lazy => 1,
);

around BUILDARGS => sub {
   my $orig = shift;
   my $self = shift;
   my %args = ref $_[0] ? %{shift()} : @_;

   die 'need a model type' unless exists $args{model};
   $args{_specified_model_type} = delete $args{model};

   $self->$orig(%args);
};

use AI::Nerl::Model::Perceptron3;

sub _build_model{
   my $self = shift;
   my $type = $self->_specified_model_type;
   die 'do model=>"Perceptron3"' unless $type eq 'Perceptron3';

   my %model_args = %{$self->_model_init_args};
   $model_args{inputs} = $self->inputs;
   $model_args{outputs} = $self->outputs;
   return AI::Nerl::Model::Perceptron3->new(%model_args);
}

# er, these go to trainer.
sub classify{
   my $self = shift;
   return $self->model->classify(@_);
}
sub spew_cost{
   my $self = shift;
   return $self->model->spew_cost(@_);
}
sub train_batch{
   my $self = shift;
   $self->model->train_batch(@_);
}

'a neural network has your dog.';
