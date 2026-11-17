"""Full KL gradient, conditioning, chunking and unchanged-state regression tests."""
from types import SimpleNamespace
import unittest
try:
    import torch
except ImportError:
    torch = None
import test_gradient_alignment as fixtures
from test_gradient_alignment import batches, REFERENCES, CANDIDATES
if torch is not None:
    from rlstack.policy.adapters.replay import row_plan
    from rlstack.runner.learners.torch_learner import full_kl_chunk


class Trunk(torch.nn.Module if torch is not None else object):
    def __init__(self, model):
        super().__init__()
        # References without registering the owning model recursively.
        object.__setattr__(self, 'owner', model)
    def forward(self, input_ids, **kw):
        model = self.owner
        h = model.embed(input_ids)
        counts = torch.arange(1, h.shape[1]+1, dtype=h.dtype)[None,:,None]
        h = h.cumsum(1)/counts
        return SimpleNamespace(last_hidden_state=h+model.proj(h))


@unittest.skipIf(torch is None, "CPU torch required for full KL gradient tests")
class KLTest(unittest.TestCase):
    direct = fixtures.GradientAlignmentTest.direct
    assertComparison = fixtures.GradientAlignmentTest.assertComparison
    enable_cache = fixtures.GradientAlignmentTest.enable_cache
    def setUp(self):
        fixtures.GradientAlignmentTest.setUp(self)
        model = self.learner._model
        model.model = Trunk(model).eval()
        model.get_output_embeddings = lambda: model.head

    def direct_kl(self, documents, prefix):
        losses, stats = [], []
        for ids, mask in documents:
            batch = batches(((ids,mask),))[0]
            with row_plan(self.learner._model).route(self.learner._rows_of(self.state,batch,1)):
                s = self.learner._model(input_ids=torch.tensor([ids])).logits[0,:-1].float()
                with torch.no_grad():
                    t = self.learner._model(input_ids=torch.tensor([tuple(prefix)+tuple(ids)])).logits[0,len(prefix):-1].float()
                    tl = t.log_softmax(-1); tp=tl.exp()
                sl=s.log_softmax(-1)
                selected=torch.tensor(mask[1:],dtype=torch.bool)
                kl=(tp*(tl-sl)).sum(-1)[selected].sum()
                nll=-sl.gather(-1,torch.tensor(ids[1:])[:,None])[:,0][selected].sum()
                losses.append(kl/selected.sum()); stats.append((float(kl.detach()),float(nll.detach())))
        grads=torch.autograd.grad(torch.stack(losses).mean(),self.params,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(self.params,grads)]),stats

    def test_kl_exact_gradient_and_independent_example_normalization(self):
        documents=tuple(CANDIDATES[0])+(((tuple(i%13 for i in range(131))), (0,)+(1,)*130),)
        prefix=(1,7)
        expected,stats=self.direct_kl(documents,prefix)
        reference,_,_=self.direct(REFERENCES)
        for budget in (5,1000):
            got=self.learner.gradient_cosines('student','pi',batches(REFERENCES,budget),
                (batches(documents,budget),),teacher_prefix_ids=prefix)[0]
            self.assertComparison(got,reference,expected.double())
            for result,(kl,nll) in zip(got.candidate_documents,stats):
                self.assertAlmostEqual(result.kl_sum,kl,places=6)
                self.assertAlmostEqual(result.nll_sum,nll,places=5)
                self.assertEqual(result.objective,'teacher_kl')
                self.assertAlmostEqual(result.kl_sum,result.student_cross_entropy_sum-result.teacher_entropy_sum,places=4)

    def test_cache_separates_sft_and_distinct_teacher_contexts(self):
        self.enable_cache()
        def run(prefix):
            return self.learner.gradient_cosines('student','pi',batches(REFERENCES),
                (batches(CANDIDATES[0]),),teacher_prefix_ids=prefix)[0]
        run(None)
        a=run((1,)); b=run((2,)); c=run((1,))
        self.assertFalse(a.candidate_cache_hit);self.assertFalse(b.candidate_cache_hit)
        self.assertTrue(c.candidate_cache_hit)
        self.assertNotEqual(a.candidate_norm,b.candidate_norm)

    def test_teacher_detached_and_full_distribution_not_sampled_targets(self):
        torch.manual_seed(4)
        head=torch.nn.Linear(3,7,bias=False)
        student=torch.randn(5,3,requires_grad=True)
        teacher=torch.randn(5,3,requires_grad=True)
        first=full_kl_chunk(head,student,teacher,torch.zeros(5,dtype=torch.long))
        second=full_kl_chunk(head,student,teacher,torch.ones(5,dtype=torch.long))
        self.assertEqual(float(first[0]),float(second[0]))
        first[0].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)

if __name__=='__main__': unittest.main()
