from jimgw.prior import *
import scipy.stats as stats


class TestUnivariatePrior:
    def test_logistic(self):
        p = LogisticDistribution(["x"])
        # Check that the log_prob is finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Cross-check log_prob with scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x))

    def test_standard_normal(self):
        p = StandardNormalDistribution(["x"])
        # Check that the log_prob is finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Cross-check log_prob with scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x))

    def test_uniform(self):
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])
        # Check that the log_prob is correct in the support
        samples = trace_prior_parent(p, [])[0].sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.allclose(log_prob, jnp.log(1.0 / (xmax - xmin)))

    def test_sine(self):
        p = SinePrior(["x"])
        # Check that the log_prob is finite
        samples = trace_prior_parent(p, [])[0].sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Check that the log_prob is correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.base_prior.transform)(x)
        y = jax.vmap(p.base_prior.transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(x), jnp.log(jnp.sin(y['x'])/2.0))
        
    def test_cosine(self):
        p = CosinePrior(["x"])
        # Check that the log_prob is finite
        samples = trace_prior_parent(p, [])[0].sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Check that the log_prob is correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(x), jnp.log(jnp.cos(y['x'])/2.0))

    def test_uniform_sphere(self):
        p = UniformSpherePrior(["x"])
        # Check that the log_prob is finite
        samples = {}
        for i in range(3):
            samples.update(trace_prior_parent(p)[i].sample(jax.random.PRNGKey(0), 10000))
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
