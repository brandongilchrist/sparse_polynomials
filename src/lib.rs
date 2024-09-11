use ark_ff::{Field, Zero, One};
use ark_test_curves::bls12_381::Fr;
use std::collections::BTreeMap;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use ark_std::rand::Rng;

static PARALLEL_THRESHOLD: AtomicU64 = AtomicU64::new(5000);

pub fn set_parallel_threshold(threshold: u64) {
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

#[derive(Clone, Debug)]
pub struct PowerCache {
    powers: Vec<Fr>,
}

// impl PowerCache {
//     pub fn new(x: &Fr, max_degree: u64) -> Self {
//         let mut powers = Vec::with_capacity(max_degree as usize + 1);
//         powers.push(Fr::one());
//         let mut current = *x;
//         for _ in 1..=max_degree {
//             powers.push(current);
//             current *= x;
//         }
//         PowerCache { powers }
//     }

impl PowerCache {
    pub fn new_2(x: &Fr, degrees: Vec<u64>) -> Self {
        let mut powers = Vec::with_capacity(degrees.len());
        // powers.push(Fr::one());
        // let mut current = *x;
        for _ in 1..= degrees.len() {
            x.pow(degrees);
           // current *= x;
        }
        PowerCache { powers }
    }
    pub fn get(&self, power: u64) -> Option<&Fr> {
        self.powers.get(power as usize)
    }
}

#[derive(Clone, Debug)]
pub struct SparsePoly {
    coeffs: BTreeMap<u64, Fr>,
    cache: Option<PowerCache>,
}

impl SparsePoly {
    pub fn new(pairs: Vec<(Fr, u64)>) -> Self {
        let mut coeffs = BTreeMap::new();
        for (coeff, index) in pairs {
            coeffs.insert(index, coeff);
        }
        SparsePoly { coeffs, cache: None }
    }

    pub fn degree(&self) -> u64 {
        self.coeffs.keys().last().cloned().unwrap_or(0)
    }

    pub fn precompute_powers(&mut self, x: &Fr) {
        let max_degree = self.degree();
        self.cache = Some(PowerCache::new(x, max_degree));
    }

    pub fn evaluate(&self, x: &Fr) -> Fr {
        if let Some(cache) = &self.cache {
            self.evaluate_with_cache(cache)
        } else {
            self.evaluate_without_cache(x)
        }
    }

    fn evaluate_with_cache(&self, cache: &PowerCache) -> Fr {
        self.coeffs
            .iter()
            .map(|(&power, coeff)| {
                *coeff * cache.get(power).expect("Power should be in cache")
            })
            .sum()
    }

    fn evaluate_without_cache(&self, x: &Fr) -> Fr {
        let mut result = Fr::zero();
        for (&degree, coeff) in &self.coeffs {
            result += *coeff * x.pow([degree]);
        }
        result
    }

    pub fn evaluate_parallel(&self, x: &Fr) -> Fr {
        let threshold = PARALLEL_THRESHOLD.load(Ordering::Relaxed);
        if self.coeffs.len() < threshold as usize {
            return self.evaluate(x);
        }

        if let Some(cache) = &self.cache {
            self.coeffs.par_iter()
                .map(|(&power, coeff)| {
                    *coeff * cache.get(power).expect("Power should be in cache")
                })
                .sum()
        } else {
            self.coeffs.par_iter()
                .map(|(&degree, coeff)| *coeff * x.pow([degree]))
                .sum()
        }
    }

    pub fn evaluate_hybrid(&self, x: &Fr) -> Fr {
        let threshold = PARALLEL_THRESHOLD.load(Ordering::Relaxed);
        
        if self.coeffs.len() < threshold as usize {
            return self.evaluate(x);
        }

        if let Some(cache) = &self.cache {
            self.coeffs.par_iter()
                .map(|(&power, coeff)| {
                    *coeff * cache.get(power).expect("Power should be in cache")
                })
                .sum()
        } else {
            self.coeffs.par_iter()
                .map(|(&degree, coeff)| *coeff * x.pow([degree]))
                .sum()
        }
    }

    pub fn mul_scalar(&mut self, scalar: &Fr) -> &mut Self {
        for coeff in self.coeffs.values_mut() {
            *coeff *= scalar;
        }
        self
    }

    pub fn div_scalar(&mut self, scalar: &Fr) -> &mut Self {
        let inv_scalar = scalar.inverse().expect("Scalar must be non-zero");
        for coeff in self.coeffs.values_mut() {
            *coeff *= inv_scalar;
        }
        self
    }

    pub fn start_chain(&self) -> ChainedOps {
        ChainedOps {
            poly: self,
            accumulated: Fr::one(),
        }
    }
}

pub struct ChainedOps<'a> {
    poly: &'a SparsePoly,
    accumulated: Fr,
}

impl<'a> ChainedOps<'a> {
    pub fn mul_scalar(mut self, scalar: &Fr) -> Self {
        self.accumulated *= scalar;
        self
    }

    pub fn div_scalar(mut self, scalar: &Fr) -> Self {
        self.accumulated *= scalar.inverse().expect("Scalar must be non-zero");
        self
    }

    pub fn evaluate(self, x: &Fr) -> Fr {
        self.poly.evaluate(x) * self.accumulated
    }

    pub fn evaluate_parallel(self, x: &Fr) -> Fr {
        self.poly.evaluate_parallel(x) * self.accumulated
    }

    pub fn evaluate_hybrid(self, x: &Fr) -> Fr {
        self.poly.evaluate_hybrid(x) * self.accumulated
    }
}

pub fn evaluate_multiple(polys: &[SparsePoly], x: &Fr) -> Vec<Fr> {
    polys.iter().map(|poly| poly.evaluate(x)).collect()
}

pub fn evaluate_multiple_parallel(polys: &[SparsePoly], x: &Fr) -> Vec<Fr> {
    polys.par_iter().map(|poly| poly.evaluate_parallel(x)).collect()
}

pub fn evaluate_multiple_hybrid(polys: &[SparsePoly], x: &Fr) -> Vec<Fr> {
    polys.par_iter().map(|poly| poly.evaluate_hybrid(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;

    fn generate_random_poly_data<R: Rng>(rng: &mut R, size: usize, sparsity: f64) -> Vec<(Fr, u64)> {
        (0..size)
            .filter_map(|i| {
                if rng.gen::<f64>() < sparsity {
                    Some((Fr::rand(rng), i as u64))
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn test_new_and_degree() {
        let poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 2),
            (Fr::from(3u64), 5),
        ]);
        assert_eq!(poly.degree(), 5);
    }

    #[test]
    fn test_evaluate() {
        let poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
            (Fr::from(3u64), 2),
        ]);
        let x = Fr::from(2u64);
        assert_eq!(poly.evaluate(&x), Fr::from(17u64));
    }

    #[test]
    fn test_evaluate_with_cache() {
        let mut poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
            (Fr::from(3u64), 2),
        ]);
        let x = Fr::from(2u64);
        poly.precompute_powers(&x);
        assert_eq!(poly.evaluate(&x), Fr::from(17u64));
    }

    #[test]
    fn test_mul_scalar() {
        let mut poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
        ]);
        poly.mul_scalar(&Fr::from(3u64));
        assert_eq!(poly.coeffs[&0], Fr::from(3u64));
        assert_eq!(poly.coeffs[&1], Fr::from(6u64));
    }

    #[test]
    fn test_div_scalar() {
        let mut poly = SparsePoly::new(vec![
            (Fr::from(2u64), 0),
            (Fr::from(4u64), 1),
        ]);
        poly.div_scalar(&Fr::from(2u64));
        assert_eq!(poly.coeffs[&0], Fr::from(1u64));
        assert_eq!(poly.coeffs[&1], Fr::from(2u64));
    }

    #[test]
    fn test_chained_ops() {
        let poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
            (Fr::from(3u64), 2),
        ]);
        let x = Fr::from(2u64);
        let result = poly.start_chain()
            .mul_scalar(&Fr::from(2u64))
            .div_scalar(&Fr::from(4u64))
            .evaluate(&x);
        assert_eq!(result, Fr::from(17u64) / Fr::from(2u64));
    }

    #[test]
    fn test_evaluate_multiple() {
        let poly1 = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
        ]);
        let poly2 = SparsePoly::new(vec![
            (Fr::from(3u64), 0),
            (Fr::from(4u64), 1),
        ]);
        let x = Fr::from(2u64);
        let results = evaluate_multiple(&[poly1, poly2], &x);
        assert_eq!(results, vec![Fr::from(5u64), Fr::from(11u64)]);
    }

    #[test]
    fn test_evaluate_hybrid() {
        let mut rng = ark_std::test_rng();
        let x = Fr::rand(&mut rng);
        
        // Small polynomial
        let small_poly = SparsePoly::new(vec![
            (Fr::from(1u64), 0),
            (Fr::from(2u64), 1),
            (Fr::from(3u64), 2),
        ]);
        assert_eq!(small_poly.evaluate(&x), small_poly.evaluate_hybrid(&x));

        // Large polynomial
        let large_poly_data = generate_random_poly_data(&mut rng, 10000, 0.01);
        let large_poly = SparsePoly::new(large_poly_data);
        assert_eq!(large_poly.evaluate(&x), large_poly.evaluate_hybrid(&x));

        // Large polynomial with cache
        let mut cached_large_poly = large_poly.clone();
        cached_large_poly.precompute_powers(&x);
        assert_eq!(cached_large_poly.evaluate(&x), cached_large_poly.evaluate_hybrid(&x));
    }

    #[test]
    fn test_evaluate_multiple_hybrid() {
        let mut rng = ark_std::test_rng();
        let x = Fr::rand(&mut rng);
        
        let polys: Vec<SparsePoly> = (0..100)
            .map(|_| {
                let poly_data = generate_random_poly_data(&mut rng, 1000, 0.01);
                SparsePoly::new(poly_data)
            })
            .collect();

        let standard_results: Vec<Fr> = polys.iter().map(|poly| poly.evaluate(&x)).collect();
        let hybrid_results = evaluate_multiple_hybrid(&polys, &x);

        assert_eq!(standard_results, hybrid_results);
    }
}