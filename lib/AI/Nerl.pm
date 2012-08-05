package AI::Nerl;
use Moose (qw'around has with' ,inner => { -as => 'moose_inner' });
use PDL;
use File::Slurp;
use JSON;

use MooseX::Storage;
use Path::Class;
with Storage(format=>'JSON'); 

# ABSTRACT: generalized machine learning?

# main_module

our $VERSION = 1.00;

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
   traits => ['DoNotSerialize'],
   lazy => 1,
);

around BUILDARGS => sub {
   my $orig = shift;
   my $self = shift;
   my %args = ref $_[0] ? %{shift()} : @_;

   #this is defined if stored, not if we're generating a new one.
   unless(defined $args{_specified_model_type}){
      die 'need a model type' unless exists $args{model};
      $args{_specified_model_type} = delete $args{model};
   }
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
sub trainer_spew_cost{
   my $self = shift;
   return $self->model->spew_cost(x=>$self->test_x, y=>$self->test_y);
}
sub train_batch{
   my $self = shift;
   $self->model->train_batch(@_);
}
sub set_test_xy{
   my ($self,$x,$y) = @_;
   $self->test_x($x);
   $self->test_y($y);
}
has test_x => (is=>'rw',isa=>'PDL',traits=>['DoNotSerialize']);
has test_y => (is=>'rw',isa=>'PDL',traits=>['DoNotSerialize']);

use PDL::IO::FlexRaw;
use File::Copy 'move';
#use pack,freeze,& write_file
sub save_to_dir{
   my ($self,$dir, %args) = @_;
   $dir = Path::Class::dir($dir) unless ref ($dir)eq'Path::Class::Dir';
   $dir = $dir->absolute->resolve;
   my $backup_dir = $dir->parent->subdir($dir->dir_list(-1,1) . '.backup');
  
   if(-e $dir){
      die 'won\'t overwrite unless you say to' unless $args{overwrite};
      $backup_dir->rmtree if -e $backup_dir;
      move($dir->stringify,$backup_dir->stringify);
   }

   $dir->mkpath;
   $self->model->save_to_dir($dir->subdir('model'));
   my $frozen = $self->freeze;
   write_file($dir->file('nerl.json')->stringify,$frozen);
   $PDL::IO::FlexRaw::writeflexhdr = 1;
   writeflex($dir->file('trainer_piddles.flex')->stringify,
      $self->test_x,$self->test_y);
}

use JSON;
sub load_from_dir{
   my $class = shift;
   my $dir = shift;
   $dir = Path::Class::dir($dir) unless ref ($dir)eq'Path::Class::Dir';
   my $frozen = read_file($dir->file('nerl.json')->stringify);
   my $from_json = from_json($frozen);
   my $model = AI::Nerl::Model::Perceptron3->load_from_dir($dir->subdir('model'));
   my %foo; #stuff that a ai::n::trainer deserves
   if(-e $dir->file('trainer_piddles.flex')){
      @foo{qw/test_x test_y/} = readflex($dir->file('trainer_piddles.flex')->stringify);
   }

   my $self = AI::Nerl->unpack($from_json, 
      inject=>{model => $model, %foo});
   return $self;
}
'a neural network has your dog.';
