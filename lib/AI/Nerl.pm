package AI::Nerl;
use Moose 'has',inner => { -as => 'moose_inner' };
use PDL;
use AI::Nerl::Network;
use File::Slurp;
use JSON;

# ABSTRACT: Neural networks with backpropagation.

# main_module

our $VERSION = .03;

#A Nerl is a mechanism to build neural networks?
#Give it training,test, and cv data?
#it settles on a learning rate and stuff?
#or maybe it's also a language for guided training?
#or maybe a visual gui thing?
#Not exactly sure. Maybe I'm tinkering with forces better left alone.
#That's a great excuse for failing horribly.


=head1 AI::Nerl - A sort of stackable neural network builder thing.

=head1 SYNOPSIS

Check out L<AI::Nerl::Network>; This module is in an early stage.

=head1 AUTHOR

Zach Morgan

=cut



has scale_input => (
   is => 'ro',
   isa => 'Num',
   required => 0,
   default => 0,
);
has [qw/inputs outputs/] => (
   is => 'ro',
   isa => 'Num'
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
   required => 0, #training can be done manually.
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


has basis => (
   is => 'ro',
   isa => 'AI::Nerl',
   required => 0,
);

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

sub save{
   my ($self,$dir) = @_;
   my $top_json = {};
   my @props = qw/l2 test_x test_y inputs outputs train_x train_y cv_x cv_y scale_input
                   network  basis/;
   die 'ugh. i dont like that nerl dir name' if $dir =~ /data|nerls$|\.|lib/;
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




'a neural network has your dog.';
