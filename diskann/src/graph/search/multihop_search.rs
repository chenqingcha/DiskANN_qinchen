/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-filtered search using multi-hop expansion.

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

use diskann_utils::Reborrow;
use diskann_utils::future::{AssertSend, SendFuture};
use diskann_vector::PreprocessedDistanceFunction;
use hashbrown::HashSet;

use super::{Knn, Search, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    graph::{
        glue::{
            self, ExpandBeam, HybridPredicate, Predicate, PredicateMut, SearchExt,
            SearchPostProcess, SearchStrategy,
        },
        index::{
            DiskANNIndex, InternalSearchStats, QueryLabelProvider, QueryVisitDecision, SearchStats,
        },
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::VectorId,
};

/// Parameters for label-filtered search using multi-hop expansion.
///
/// This search extends standard graph search by expanding through non-matching
/// nodes to find matching neighbors. More efficient than flat search when the
/// matching subset is reasonably large.
#[derive(Debug)]
pub struct MultihopSearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining node matches.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<'q, InternalId> MultihopSearch<'q, InternalId> {
    /// Create new multihop search parameters.
    pub fn new(inner: Knn, label_evaluator: &'q dyn QueryLabelProvider<InternalId>) -> Self {
        Self {
            inner,
            label_evaluator,
        }
    }
}

impl<'q, DP, S, T, O, OB> Search<DP, S, T, O, OB> for MultihopSearch<'q, DP::InternalId>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
{
    type Output = SearchStats;

    fn search(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), start_ids.len());

            let stats = multihop_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                self.label_evaluator,
            )
            .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(self.inner.l_value().get()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

/// A predicate that checks if an item is not in the visited set AND matches the label filter.
///
/// Used during two-hop expansion to filter neighbors based on both visitation
/// status and label matching criteria.
pub struct NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    visited_set: &'a mut HashSet<K>,
    query_label_evaluator: &'a dyn QueryLabelProvider<K>,
}

impl<'a, K> NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    /// Construct a new `NotInMutWithLabelCheck` around `visited_set`.
    pub fn new(
        visited_set: &'a mut HashSet<K>,
        query_label_evaluator: &'a dyn QueryLabelProvider<K>,
    ) -> Self {
        Self {
            visited_set,
            query_label_evaluator,
        }
    }
}

impl<K> Predicate<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval(&self, item: &K) -> bool {
        !self.visited_set.contains(item) && self.query_label_evaluator.is_match(*item)
    }
}

impl<K> PredicateMut<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval_mut(&mut self, item: &K) -> bool {
        if self.query_label_evaluator.is_match(*item) {
            return self.visited_set.insert(*item);
        }
        false
    }
}

impl<K> HybridPredicate<K> for NotInMutWithLabelCheck<'_, K> where K: VectorId {}

// ---------------------------------------------------------------------------
// Exploration Queue support types
// ---------------------------------------------------------------------------

/// A wrapper around [`Neighbor`] that reverses the natural ordering so that a
/// [`BinaryHeap`] yields the **closest** (smallest distance) entry via `pop()`.
///
/// Rust's `BinaryHeap` is a max-heap, so we store entries in *descending*
/// distance order. Popping therefore gives us the entry with the smallest
/// distance — greedy best-first behaviour.
#[derive(Debug, Clone, Copy)]
struct ExplorationEntry<I: VectorId> {
    neighbor: Neighbor<I>,
}

impl<I: VectorId> PartialEq for ExplorationEntry<I> {
    fn eq(&self, other: &Self) -> bool {
        self.neighbor.distance == other.neighbor.distance
    }
}

impl<I: VectorId> Eq for ExplorationEntry<I> {}

impl<I: VectorId> PartialOrd for ExplorationEntry<I> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl<I: VectorId> Ord for ExplorationEntry<I> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse: smaller distance = higher priority in the max-heap.
        other
            .neighbor
            .distance
            .partial_cmp(&self.neighbor.distance)
            .unwrap_or(CmpOrdering::Equal)
    }
}

/// Internal multihop search implementation.
///
/// Performs label-filtered search by expanding through non-matching nodes
/// to find matching neighbors within two hops.
pub(crate) async fn multihop_search_internal<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    query_label_evaluator: &dyn QueryLabelProvider<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    T: ?Sized,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();

    // Helper to build the final stats from scratch state.
    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialize search state if not already initialized.
    // This allows paged search to call multihop_search_internal multiple times
    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;

        for id in start_ids {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            scratch.best.insert(Neighbor::new(id, dist));
        }
    }

    // Pre-allocate with good capacity to avoid repeated allocations
    let mut one_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut two_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut candidates_two_hop_expansion = Vec::with_capacity(max_degree_with_slack);

    // --- Exploration queue state ---
    // Capacity is l_search; the queue is only populated when the label
    // provider signals low match rate via RejectAndNeedExpand.
    let l_search = search_params.l_value().get();
    let mut exploration_queue: BinaryHeap<ExplorationEntry<I>> =
        BinaryHeap::with_capacity(l_search);
    let mut exploration_set: HashSet<I> = HashSet::with_capacity(l_search);
    let mut need_expand_active = false;

    loop {
        let has_best = scratch.best.has_notvisited_node();
        let has_exploration = !exploration_queue.is_empty();

        // Terminate when neither source has candidates.
        if !has_best && !has_exploration {
            break;
        }
        if accessor.terminate_early() {
            break;
        }

        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();
        candidates_two_hop_expansion.clear();
        two_hop_neighbors.clear();

        // --- Beam fill (priority: matching nodes from scratch.best first) ---
        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        // Fill remaining beam slots from the exploration queue when active.
        if need_expand_active {
            while scratch.beam_nodes.len() < beam_width {
                if let Some(entry) = exploration_queue.pop() {
                    scratch.beam_nodes.push(entry.neighbor.id);
                } else {
                    break;
                }
            }
        }

        // Nothing to expand this iteration — should not happen given the
        // outer loop guard, but be safe.
        if scratch.beam_nodes.is_empty() {
            break;
        }

        // compute distances from query to one-hop neighbors, and mark them visited
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| one_hop_neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for neighbor in one_hop_neighbors.iter().copied() {
            match query_label_evaluator.on_visit(neighbor) {
                QueryVisitDecision::Accept(accepted) => {
                    scratch.best.insert(accepted);
                }
                QueryVisitDecision::Reject => {
                    // Rejected nodes: still add to two-hop expansion so we can traverse through them
                    candidates_two_hop_expansion.push(neighbor);
                }
                QueryVisitDecision::RejectAndNeedExpand => {
                    // Low match rate detected — enable exploration queue.
                    need_expand_active = true;
                    candidates_two_hop_expansion.push(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    return Ok(make_stats(scratch));
                }
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // sort the candidates for two-hop expansion by distance to query point
        candidates_two_hop_expansion.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // limit the number of two-hop candidates to avoid too many expansions
        candidates_two_hop_expansion.truncate(max_degree_with_slack / 2);

        // Expand each two-hop candidate: if its neighbor is a match, compute its distance
        // to the query and insert into `scratch.visited`
        // If it is not a match, do nothing
        let two_hop_expansion_candidate_ids: Vec<I> =
            candidates_two_hop_expansion.iter().map(|n| n.id).collect();

        accessor
            .expand_beam(
                two_hop_expansion_candidate_ids.iter().copied(),
                computer,
                NotInMutWithLabelCheck::new(&mut scratch.visited, query_label_evaluator),
                |distance, id| {
                    two_hop_neighbors.push(Neighbor::new(id, distance));
                },
            )
            .await?;

        // Next, insert the new matches into `scratch.best` and increment stats counters
        two_hop_neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += two_hop_neighbors.len() as u32;
        scratch.hops += two_hop_expansion_candidate_ids.len() as u32;

        // --- Feed the exploration queue ---
        // After two-hop expansion, the rejected one-hop candidates have been
        // used for their two-hop reach. Push them into the exploration queue
        // so the search can continue expanding the graph even when
        // scratch.best has no more unvisited matching nodes.
        if need_expand_active {
            for candidate in &candidates_two_hop_expansion {
                if exploration_set.insert(candidate.id) {
                    exploration_queue.push(ExplorationEntry {
                        neighbor: *candidate,
                    });
                    // Enforce capacity limit: drop the farthest entry.
                    if exploration_queue.len() > l_search {
                        // The heap is min-by-distance (reversed), so the
                        // *last* element in the internal vec is the farthest.
                        // BinaryHeap doesn't expose that directly, but since
                        // we only exceed by 1 we can just let it grow by one
                        // and it will naturally be displaced next iteration.
                        // For a tighter bound we drain:
                        while exploration_queue.len() > l_search {
                            exploration_queue.pop();
                        }
                    }
                }
            }
        }
    }

    Ok(make_stats(scratch))
}
