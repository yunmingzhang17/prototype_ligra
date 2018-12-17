#include "ligra.h"
#include "index_map.h"
#include "bucket.h"
#include "edgeMapReduce.h"
#include <mm_malloc.h>


//#define DEBUG
//#define PROFILE
//#define TIME


int Check_Bit(unsigned int *Vis, long long int delta)
{
  // Function returns 0 if the bit was not set... 1 otherwise...

  long long int word = ((delta) >> 5);
  int bit = ( (delta) & 0x1F); // bit will be [0..31]
  // [profile] loading Vis[word] takes 31.1% of memory stall time.
  // stall per load = 73.4 ns.
  unsigned int value = Vis[word];
  //printf("addr = %#lx\n", &(Vis[word]));
  if ((value & (1 << bit)) != 0) return 1;
  return 0;
}

int Set_Bit_Only(unsigned int *Vis, long long int delta)
{
  // Function returns 0 if the bit was not set... 1 otherwise and  also sets the bit :)

  long long int word = (delta >> 5);
  int bit = ( delta & 0x1F); // bit will be [0..31]
  // [profile] loading Vis[word] takes 2.3% stall time
  // stall per load = 115.9 ns.
  unsigned int value = Vis[word];

  if ((value & (1 << bit)) != 0) return 1;
  // [profile] storing Vis[word] takes 2.55% stall time
  // stall per store = 128.5 ns.
  Vis[word] =  (value | (1<<bit));
  return 0;
}



template <class vertex>
array_imap<uintE> KCore(graph<vertex>& GA, size_t num_buckets=128) {

  #ifdef PROFILE
  timer t1;
  timer t2;
  timer t0;
  t0.start();
  timer t_setup;
  t_setup.start();
  timer t_next_bucket;
  #endif

  #ifdef TIME
  timer t_iter;
  #endif

  timer t_D;
  timer t_em;
  timer t_b;
  
  const size_t n = GA.n; const size_t m = GA.m;
  t_D.start();
  auto D = array_imap<uintE>(n, [&] (size_t i) { return GA.V[i].getOutDegree(); });
  t_D.stop();

  t_em.start();
  auto em = EdgeMap<uintE, vertex>(GA, make_tuple(UINT_E_MAX, 0), (size_t)GA.m/5);
  t_em.stop();

  t_b.start();
  auto b = make_buckets(n, D, increasing, num_buckets);
  t_b.stop();

  #ifdef PROFILE
  t_setup.stop();
  t_setup.reportTotal("set up took: " );
  t_D.reportTotal("t_D: ");
  t_em.reportTotal("t_em: ");
  t_b.reportTotal("t_b: ");
  #endif

  size_t finished = 0;

  #ifdef DEBUG
  int round = 0;
  #endif

  int check_bit_count = 0;
  int pass_bit_check_count = 0;

  int bit_vector_length = n/ 32;
  if(n % 32 != 0)
    bit_vector_length++;
  unsigned int* vertexSubsetBit = (unsigned int*) _mm_malloc(sizeof(unsigned int) * bit_vector_length, 64);
  
  // reset all bits of the bit vector
  for(int i = 0; i < bit_vector_length; i++) 
    vertexSubsetBit[i] = 0;


  while (finished != n) {
    
    #ifdef TIME
    t_iter.start();
    #endif

    #ifdef DEBUG
    round++;
    //std::cout << "round: " << round << std::endl;
    #endif

    #ifdef PROFILE
    t_next_bucket.start();
    #endif

    auto bkt = b.next_bucket();

    #ifdef PROFILE
    t_next_bucket.stop();
    #endif

    auto active = bkt.identifiers;
    uintE k = bkt.id;
    finished += active.size();

    auto apply_f = [&] (const tuple<uintE, uintE>& p) -> const Maybe<tuple<uintE, uintE> > {
      uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);

#ifdef COUNT
      check_bit_count++;
#endif
      if (!Check_Bit(vertexSubsetBit, v)){

#ifdef COUNT
	pass_bit_check_count++;
#endif

	uintE deg = D.s[v];
	if (deg > k) {
	  uintE new_deg = max(deg - edgesRemoved, k);
	  if (new_deg == k) { //belongs to the current core, shouldn't be in the next round
	    Set_Bit_Only(vertexSubsetBit, v);
	  }
	  D.s[v] = new_deg;
	  uintE bkt = b.get_bucket(deg, new_deg);
	  return wrap(v, bkt);
	}

      }// end of Check_Bit if statement
	return Maybe<tuple<uintE, uintE> >();
      };

#ifdef PROFILE
      t1.start();
#endif
      
      //std::cout << "active vertices count: " << active.size() << std::endl;
      vertexSubsetData<uintE> moved = em.template edgeMapCount<uintE>(active, apply_f);
      //std::cout << "updated vertices: " << moved.m  << std::endl;
      

#ifdef PROFILE
      t1.stop();
#endif

#ifdef PROFILE
      t2.start();
#endif

    b.update_buckets(moved.get_fn_repr(), moved.size());

#ifdef PROFILE
    t2.stop();
#endif


  #ifdef TIME
    std::cout << "time for this iter: " <<  t_iter.stop() << " round: " << round << std::endl;
    
  #endif

    moved.del(); active.del();
  }//end of while loop

#ifdef PROFILE
  t0.stop();
  t0.reportTotal("total time: ");

  t1.reportTotal("edgeMapCount took: ");
  t2.reportTotal("update_buckets took: ");
  t_next_bucket.reportTotal("move to the next bucket took: ");

  std::cout << "check bit count: " << check_bit_count << std::endl;
  std::cout << "pass check bit count: " << pass_bit_check_count << std::endl;

#endif

#ifdef DEBUG
    std::cout << "round: " << round << std::endl;
#endif



  return D;
}

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) {
  size_t num_buckets = P.getOptionLongValue("-nb", 128);
  if (num_buckets != (1 << pbbs::log2_up(num_buckets))) {
    cout << "Number of buckets must be a power of two." << endl;
    exit(-1);
  }
  cout << "### Application: k-core" << endl;
  cout << "### Graph: " << P.getArgument(0) << endl;
  cout << "### Workers: " << getWorkers() << endl;
  cout << "### Buckets: " << num_buckets << endl;
  cout << "### n: " << GA.n << endl;
  cout << "### m: " << GA.m << endl;

  auto cores = KCore(GA, num_buckets);
  uintE mc = 0;
  int sum_core = 0;
  for (size_t i=0; i < GA.n; i++) { mc = std::max(mc, cores[i]); sum_core += cores[i]; }
  cout << "### Max core: " << mc << endl;
  cout << "### sum of core: " << sum_core << endl;
}
