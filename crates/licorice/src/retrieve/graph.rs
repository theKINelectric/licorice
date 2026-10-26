//! Bounded graph expansion, retaining the connecting path for atomic packing.
use crate::{Result, Store};
use rusqlite::params;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug, Serialize)]
pub struct GraphPath {
    pub target: i64,
    pub facts: Vec<i64>,
}
pub(super) async fn expand(store: &Store, seeds: &[i64], hops: usize, limit: usize) -> Result<HashMap<i64, Vec<i64>>> {
    let seeds = seeds.to_vec();
    store.read(move |c| {
  let mut frontier:Vec<_>=seeds.iter().take(16).map(|id|(*id,Vec::<i64>::new())).collect();
  let mut visited:HashSet<i64>=seeds.iter().copied().collect();let mut paths=HashMap::new();
  for _ in 0..hops {
   let mut next=Vec::new();
   for (entity,path) in frontier {
    let mut s=c.prepare("SELECT id,subject_id,object_id FROM facts WHERE (subject_id=?1 OR object_id=?1) AND status!='retracted' ORDER BY recorded_last DESC,id LIMIT 16")?;
    let rows=s.query_map(params![entity],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<i64>>(2)?)))?;
    for row in rows {
     let (id,subject,object)=row?;
     if paths.len()>=limit {return Ok(paths);}
     if paths.contains_key(&id){continue;}
     let mut link=path.clone();link.push(id);paths.insert(id,link.clone());
     let other=if subject==entity {object}else{Some(subject)};
     if let Some(other)=other {if visited.len()<32 && visited.insert(other){next.push((other,link));}}
    }
   }
   frontier=next;
  }
  Ok(paths)
 }).await
}
