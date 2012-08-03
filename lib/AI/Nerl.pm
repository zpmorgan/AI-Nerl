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

1;
__END__

#initialize $self->network, but don't train.
# any parameters AI::Nerl::Network takes are fine here.
sub init_network{
   my $self = shift;
   my %nn_params = @_;
   #input layer size:
   unless ($nn_params{l1}){
      if ($self->basis){
         $nn_params{l1} = $self->basis->network->l1 + $self->basis->network->l2;
      } elsif($self->train_x) {
         $nn_params{l1} ||= $self->train_x->dim(1);
      }
   }
   #output layer size:
   unless ($nn_params{l3}){
      if ($self->basis){
         $nn_params{l3} =  $self->basis->network->l3;
      } elsif($self->train_x) {
         $nn_params{l3} ||= $self->train_y->dim(1);
      }
   }
   $nn_params{l2} ||= $self->l2;
   $nn_params{scale_input} ||= $self->scale_input;

   my $nn = AI::Nerl::Network->new(
      %nn_params
   );
   $self->network($nn);
}

sub resize_l2{
   my $self = shift;
   my $new_l2 = shift;
   $self->l2($new_l2);
   $self->network->resize_l2($new_l2);
}

sub init{
   my $self = shift;
   $self->build_network();
}

sub build_network{
   my $self = shift;
   my $l1 = $self->inputs // $self->test_x->dim(1);
   my $l3 = $self->outputs // $self->test_y->dim(1);
   my $nn = AI::Nerl::Network->new(
      l1 => $l1,
      l2 => $self->l2,
      l3 => $l3,
      scale_input => $self->scale_input,
   );
   $self->network($nn);
}

sub append_l2{
   my ($self,$x) = @_;
   if($self->basis){
      $x = $self->basis->append_l2($x);
   }
   return $self->network->append_l2($x);
}


sub run{
   my ($self,$x) = @_;
   $x->sever;
   if($self->basis){
      $x = $self->basis->append_l2($x);
   }
   return $self->network->run($x);
}
sub train{
   my ($self,$x,$y) = @_;
   $x->sever;
   if($self->basis){
      $x = $self->basis->append_l2($x);
   }
   return $self->network->train($x,$y);
}

sub cost{
   my ($self,$x,$y) = @_;
   unless ($x and $y){
      $x = $self->test_x->copy();
      $y = $self->test_y->copy();
   }
   $x->sever();
   if($self->basis){
      $x = $self->basis->append_l2($x);
   }
   return $self->network->cost($x,$y);
}

# A nerl occupied a directory.
# its parameters occupy a .json file.
# its piddles occupy .fits files.
# Its network(s) occupy subdirectories,
#  in which network piddles occupy .fits files
#  and network params occupy another .json file.

use PDL::IO::FITS;
use File::Path;

my @props = qw/l2 test_x test_y inputs outputs train_x train_y cv_x cv_y scale_input
                   network  basis/;

sub save{
   my ($self,$dir) = @_;
   my $top_json = {};
   #die 'ugh. i dont like that nerl dir name' if $dir =~ /data|nerls$|\.|lib/;
   rmtree $dir if -d $dir;
   mkdir $dir;
   for my $p (@props){
      next unless defined $self->$p;
      $top_json->{$p} = $self->$p;
      if (ref $top_json->{$p} eq 'PDL'){
         my $pfile = "$p.fits";
         #switcharoo with file name
         $top_json->{$p}->wfits("$dir/$pfile");
         $top_json->{$p} = $pfile;
      }
      elsif (ref $top_json->{$p} eq 'AI::Nerl::Network'){
         my $nn = $self->$p;
         my $nndir = "$dir/$p";
         $top_json->{$p} = "|AINN|$nndir";
         $nn->save($nndir);
      }
   }
   my $encoded_nerl = to_json($top_json);
   write_file("$dir/attribs", $encoded_nerl);
}

sub load{
   my $dir = shift;
   $dir = shift if $dir eq 'AI::Nerl';
   die 'symptom ai::nerl->load(lack of dir?)' unless $dir;
   my $from_json = from_json(read_file("$dir/attribs"));
   my %to_nerl;
   for my $a (@props){
      my $value = $from_json->{$a};
      next unless defined $value;
      $to_nerl{$a} = $value;
      #special cases: ai::nerl::networks and piddles
      if ($value =~ /\.fits$/){
         my $piddle = rfits("$dir/$value");
         $to_nerl{$a} = $piddle;
      }
      elsif ($value =~ /^\|AINN\|(.*)$/){ #load a AI::N::network
         my $nn = AI::Nerl::Network->load($1);
         $to_nerl{$a} = $nn;
      }
   }
   my $self = AI::Nerl->new(%to_nerl);
   return $self;
}

'a neural network has your dog.';
